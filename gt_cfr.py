import pyspiel
import numpy as np
from collections import OrderedDict

from state_structure import StateStructure, PyspielStateStructure
from regret_minimizer import *

STORE_HISTORY = True

# game node keys
PLAYER_SEQUENCES = 'player_sequences'
CHANCE_OUTCOMES = 'chance_outcomes'
RETURNS = 'returns'
EVALUATION = 'evaluation'

# single player dict keys
PARENT_SEQUENCE = 'parent_sequence'
LEGAL_ACTIONS = 'legal_actions'
INFOSET = 'infoset'
COND_VALS = 'cond_vals'
COND_VAR_SUMS = 'cond_vars'
VISIT_CT = 'visit_ct'
CHILD_VISIT_CT = 'child_visit_ct'
CHILD_VISIT_WEIGHTED_CT = 'child_visit_wt_ct'


class GameNode:
    def __init__(self,
                 parent,
                 player,
                 infoset_id,
                 reach_prob_chance,
                 legal_actions,
                 children=None,
                 last_action=None,
                 terminal=False,
                 **kwargs):
        self.parent = parent
        self.player = player
        self.last_action = last_action
        self.reach_prob_chance = reach_prob_chance
        self.infoset_id = infoset_id
        self.legal_actions = legal_actions
        self.children = children
        if self.children is None:
            self.children = dict()
        self.terminal = terminal
        self.data = kwargs

    def get_parent_sequence(self, player=None):
        """
        gets last decision point of player in ancestors, not including self
            if player is unspecified, uses self.player
        """
        if self.parent is None:
            return None
        if player is None:
            player = self.player

        def parent_seq_helper(node, action):
            """
            gets parent sequence by climbing up tree
            :param node: ancestor node
            :param action: action played in history after ancestor
            """
            if node is None:
                return None
            if node.player == player:
                return node.infoset_id, action
            else:
                return parent_seq_helper(node.parent, node.last_action)

        return parent_seq_helper(node=self.parent, action=self.last_action)

    def update_data(self, **kwargs):
        self.data.update(kwargs)

    def is_chance_node(self):
        return (not self.terminal) and self.player < 0

    def get_history(self):
        if self.parent is None:
            return ()
        else:
            return self.parent.get_history() + (self.last_action,)

    def get_full_player_sequence(self, player):
        """
        get all (infoset id, action) along the path to this node for player
        :param player: player to query
        :return:
        """
        # if this key exists, we can just grab it from here
        if PLAYER_SEQUENCES in self.data and player in self.data[PLAYER_SEQUENCES]:
            return self.data[PLAYER_SEQUENCES][player]
        if self.parent is None:
            return ()
        if self.parent.player == player:
            return self.parent.get_full_player_sequence(player) + ((self.parent.infoset_id, self.last_action),)
        else:
            return self.parent.get_full_player_sequence(player)


class GTCFR:
    def __init__(self, root_state: StateStructure,
                 rm_class: type(RegretMinimizer) = RegretMatchingPlus,
                 rm_kwargs=None,
                 ):
        self.root_state = root_state
        self.rm_class = rm_class
        self.rm_kwargs = rm_kwargs
        if self.rm_kwargs is None:
            self.rm_kwargs = dict()

        self.root = GameNode(parent=None,
                             player=root_state.current_player(),
                             terminal=root_state.is_terminal(),
                             infoset_id=root_state.get_infoset_id(),
                             reach_prob_chance=1.,
                             legal_actions=root_state.legal_actions(),

                             **{RETURNS: root_state.returns()}
                             )
        if self.root.is_chance_node():
            self.root.update_data(**{CHANCE_OUTCOMES: self.root_state.chance_outcomes()})

        # dict of single_player_trees -> tree
        # tree is an ORDERED DICT of infoset ids in topdown order
        # (infoset id)-> DICT(
        #                   PARENT_SEQUENCE: parent sequence,
        #                   LEGAL_ACTIONS: legal actions,
        #                   INFOSET: all nodes in infoset,
        #                   CF_VALS: (action -> estimated counterfactual value),
        #                   VISIT_CT: times visited in expansion sampling,
        #                   CHILD_VISIT_CT: (action -> times the child is visited in expansion sampling),
        #                   )
        self.single_player_trees = dict()
        self.add_to_extra_structrues(self.root)

        # dict of player -> (infoset id -> regret minimizer)
        self.player_to_regret_minimizers = dict()
        self.maybe_add_regret_minimizer(player=self.root.player,
                                        infoset_id=self.root.infoset_id,
                                        actions=self.root.legal_actions,
                                        )

    def make_leaf(self, node, state, action):
        assert action not in node.children
        state_prime = state.child(action)
        action_prob_chance = 1.
        if node.is_chance_node():
            action_prob_chance = node.data[CHANCE_OUTCOMES][action]
        leaf = GameNode(
            parent=node,
            player=state_prime.current_player(),
            last_action=action,
            terminal=state_prime.is_terminal(),
            infoset_id=state_prime.get_infoset_id(),
            reach_prob_chance=node.reach_prob_chance*action_prob_chance,
            legal_actions=state_prime.legal_actions(),
        )
        child_player_sequences = node.data.get(PLAYER_SEQUENCES, dict())
        if not node.is_chance_node():
            child_player_sequences = child_player_sequences.copy()  # can just send the same object to chance nodes
            child_player_sequences[node.player] = child_player_sequences.get(node.player, ()) + (
                (node.infoset_id, action),)

        leaf.update_data(**{PLAYER_SEQUENCES: child_player_sequences})
        if leaf.terminal:
            leaf.update_data(returns=state_prime.returns())
        if leaf.is_chance_node():
            leaf.update_data(**{CHANCE_OUTCOMES: state_prime.chance_outcomes()})
        node.children[action] = leaf
        if len(node.children) == len(node.legal_actions) and EVALUATION in node.data:
            # clear evaluation from pre-terminal nodes
            node.data.pop(EVALUATION)
        self.maybe_add_regret_minimizer(player=leaf.player,
                                        infoset_id=leaf.infoset_id,
                                        actions=leaf.legal_actions,
                                        )
        self.add_to_extra_structrues(leaf)
        return leaf

    def add_to_extra_structrues(self, node, **kwargs):
        """
        adds to extra structures, including tfsdp for player (if applicable)
        :param node:
        :return:
        """
        if not node.is_chance_node() and not node.terminal:
            # add to tfsdp structure
            if node.player not in self.single_player_trees:
                self.single_player_trees[node.player] = OrderedDict()
            # reasigning a value does not change order, an infoset will be ordered based on the FIRST time it is seen
            # though doing this just to be safe
            if node.infoset_id not in self.single_player_trees[node.player]:
                self.single_player_trees[node.player][node.infoset_id] = {PARENT_SEQUENCE: node.get_parent_sequence(),
                                                                          LEGAL_ACTIONS: node.legal_actions,
                                                                          INFOSET: set(),
                                                                          COND_VALS: dict(),
                                                                          COND_VAR_SUMS: dict(),
                                                                          VISIT_CT: 0,
                                                                          CHILD_VISIT_CT: {a: 0 for a in
                                                                                           node.legal_actions},
                                                                          CHILD_VISIT_WEIGHTED_CT: {a: 0 for a in
                                                                                                    node.legal_actions},
                                                                          }
            # add game tree node to infoset
            self.single_player_trees[node.player][node.infoset_id][INFOSET].add(node)

    def maybe_add_regret_minimizer(self, player, infoset_id, actions, force_reset=False):
        if self.rm_class is None:
            return
        if player >= 0:
            if player not in self.player_to_regret_minimizers:
                self.player_to_regret_minimizers[player] = dict()

            # if we have not seen this before, or if we want to reset the regret minimizer
            if force_reset or infoset_id not in self.player_to_regret_minimizers[player]:
                rm = self.rm_class(action_set=actions, **self.rm_kwargs)
                self.player_to_regret_minimizers[player][infoset_id] = rm
                return rm

    def reset_regret_minimizers(self,warm_start=None):
        for p, dic in self.player_to_regret_minimizers.items():
            for infoset_id, regret_min in dic.items():
                regret_min.reset(warm_start=warm_start)

    def evaluate_and_push(self, players, node, player_strategies, sequential_form=False):
        """
        evaluats a leaf node (node is either terminal or should be a leaf of the tree (newly added))
        :param players: players to back up node value for
        :param node: node to evaluate
        :param player_strategies player -> strategy they used in sampling of node. only needs to be defined for players in players
        :return:
        """

        state = self.state_of(node)
        evaluation = state.evaluate()
        node.update_data(**{EVALUATION: evaluation})
        for player in players:
            parent_seq = node.get_parent_sequence(player=player)
            if parent_seq is not None:
                infoset_id, parent_action = parent_seq
                self.backprop_terminal_value(player=player,
                                             infoset_id=infoset_id,
                                             action_taken=parent_action,
                                             terminal_utility=evaluation[player],
                                             player_strategy=player_strategies[player],
                                             sequential_form=sequential_form,
                                             )

    def backprop_terminal_value(self, player, infoset_id, action_taken, terminal_utility, player_strategy,
                                sequential_form=False):
        """
        terminal_CF_utility is the utility for player collected at the sample

        :param player:
        :param infoset_id:
        :param terminal_utility: utility for player collected at the sampled terminal (or leaf) node
            this is u(z)
                NOTE z is sampled with probability r(z)y(z)p(z)
                    where r is the reach probability of expanding player (using reference strategy)
                    y is reach prob of opponent
                    p is reach prob of chance
                We want to find conditional_value(x,y,j,a); infoset j is a collection of histories (game nodes)
                cf_value(x,y,j,a)=sum_{h in j; z desc. (h,a)}u(z)x(z|(h,a))y(z)p(z)
                conditional value is cf_value normalized by external reach probability of j
                conditional_value(x,y,j,a)=cf_value(x,y,j,a)/(sum_{h in j}y(h)p(h))

                If we take the count (sum of 1) for every ancestor infoset of z, the value of this stored at j is (over k trials)
                    K sum_{h in j}r(h)y(h)p(h)
                Weighting samples by 1/r(h) yields
                    K sum_{h in j}y(h)p(h)
                counting (j,a) instead of j yields
                    A := K sum_{h in j}r(h,a|h)y(h)p(h)
                        = K r(j,a|j) sum_{h in j}y(h)p(h)
                if we ignore weighting:
                    A' := K r(j,a) sum_{h in j}y(h)p(h)

                Taking the sum (over K trials) of u(z)x(z|h)/r(z) yields (if x=r, this is u(z)/r(h))
                    B := K x(j,a|j) sum_{h in j; z desc. (h,a)}u(z)x(z|h,a)y(z)p(z) = K x(j,a|j) cf_value(x,y,j,a)
                Ignoring the weighting and assuming x=r, we get
                    B' := K r(j,a) cf_value(x,y,j,a)
                then dividing B/A or B'/A' yields cf_value(x,y,j,a)/sum_{h in j}y(h)p(h), the conditional value
                    approximating x=r, this is equivalent to a weighted mean, weighting each sample by 1/r(h)
        :param player_bhv_strategies:
        :return:
        """
        if not sequential_form:
            player_strategy = self.convert_to_sequence_form(player=player, behavioral_strat=player_strategy)
            sequential_form = True
        infoset_dic = self.single_player_trees[player][infoset_id]
        parent_seq = infoset_dic[PARENT_SEQUENCE]

        infoset_plyr_reach_prob = 1.
        if parent_seq is not None:
            # if not the first encountered decision point, the reach probability for player is the
            infoset_plyr_reach_prob = player_strategy[parent_seq]

        # UPDATE RUNNING MEAN OF COND VALUES
        #  (Welford's method, follow along https://uditsamani.com/welford/, with weighted version)
        #  data is weighted, so mu_n=sum_n(wixi)/sum_n(wi); S_n = sum_{n}wi(xi-mu_n)^2
        #  mean update ends up being mu_{n+1}=mu_n + w_{n+1}(x_{n+1}-mu_n)/{sum_{n+1} wi}
        # IGNORE the fact that x is not r, call it good enough
        if False:
            w = 1/infoset_plyr_reach_prob
        else:
            w = 1

        old_weight = infoset_dic[CHILD_VISIT_WEIGHTED_CT].get(action_taken, 0)
        new_weight = old_weight + w
        old_cond_val = infoset_dic[COND_VALS].get(action_taken, 0.)

        delta = terminal_utility - old_cond_val
        # infoset_dic[COND_VALS][action_taken] = (old_weight*old_cond_val + terminal_utility/infoset_plyr_reach_prob)/(new_weight)
        new_cond_val = old_cond_val + w*delta/new_weight
        infoset_dic[COND_VALS][action_taken] = new_cond_val

        # UPDATE RUNNING VAR OF COND VALUES

        #  variance update ends up being S_{n+1} = S_n + w_{n+1}(x_{n+1}-mu_n)(x_{n+1}-mu_{n+1})
        #  variance is S_{n+1}/sum_n{wi}, so need to store the old weight as well
        old_cond_var_sum, _ = infoset_dic[COND_VAR_SUMS].get(action_taken, (0., 0.))
        new_cond_var_sum = old_cond_var_sum + w*delta*(terminal_utility - new_cond_val)
        infoset_dic[COND_VAR_SUMS][action_taken] = (new_cond_var_sum, old_weight)

        # UPDATE WEIGHTS, VISIT COUNT, CHILD VISIT CT
        infoset_dic[CHILD_VISIT_WEIGHTED_CT][action_taken] = new_weight
        infoset_dic[VISIT_CT] += 1
        infoset_dic[CHILD_VISIT_CT][action_taken] += 1
        if parent_seq is not None:
            parent_infoset_id, parent_action_taken = parent_seq
            self.backprop_terminal_value(player=player,
                                         infoset_id=parent_infoset_id,
                                         action_taken=parent_action_taken,
                                         terminal_utility=terminal_utility,
                                         player_strategy=player_strategy,
                                         sequential_form=sequential_form,
                                         )

    def sample_leaf_spot(self, expanding_players, player_bhv_strategies, p=0.5):
        """
        samples a node, action that is outside of the tree, using the strategies given
            ensures that either node is terminal (in which case action is None)
            or node,action leads outside the current tree
        :param expanding_players: the players that are using an expanding policy (with probability p, take action at j with max Q(j,a))
            Q(j,a)=mean_value(x,y|j,a)+C*var_value(x,y|j,a)*sqrt(N(j))/(1+N(j,a))
            value is accumulated over all EXPANSION phases instead of during the CFR updates
            value of action a from j is specifically mean of: (estimated CFR value of descendants of j)/(opponent reach probability summed over j)
        :param player_bhv_strategies: for each player i, other_player_strategies[i][infoset id of i] is a policy for this infoset
            IS BEHAVIORAL,
        :return: (node,action)
        """
        if expanding_players is None:
            expanding_players = set()
        node = self.root
        action = None
        while True:
            actions = list(node.legal_actions)
            if node.terminal:
                action = None
                break
            if node.is_chance_node():
                dist = [node.data[CHANCE_OUTCOMES][a] for a in actions]
            else:
                strat = player_bhv_strategies[node.player][node.infoset_id]
                dist = np.array([strat[a] for a in actions])
                if node.player in expanding_players:
                    # distribution should be (1-p)*{dist proportional to 1 for support of player strategy} + p*{1 for max of PUCT selection}

                    action = self.PUCT_selection(node)
                    action_idx = actions.index(action)

                    support = (dist > 0)
                    dist = (1 - p)*support/np.sum(
                        support)  # distribution where the player strategy is positive, sums to (1-p)
                    dist[action_idx] += p  # add p to the action selected by PUCT
            action = np.random.choice(actions, p=dist)
            if action not in node.children:
                # this is outside the tree
                break
            else:
                # recurse on child
                node = node.children[action]
        return (node, action)

    def PUCT_selection(self, node):
        """
        select an action for a non-terminal infoset
        :param node:
        :return: action selected by PUCT
        """
        player = node.player
        infoset_dic = self.single_player_trees[player][node.infoset_id]
        Nj = infoset_dic[VISIT_CT]
        C = 1.
        actions = list(infoset_dic[LEGAL_ACTIONS])
        Q = []
        for action in actions:
            Nja = infoset_dic[CHILD_VISIT_CT][action]
            S, W = infoset_dic[COND_VAR_SUMS].get(action, (0., 0.))
            if W == 0 or S/W <= 1E-3:
                var = 1
            else:
                var = S/W
            mu = infoset_dic[COND_VALS].get(action, 0.)
            Q.append(mu + C*var*np.sqrt(Nj)/(1 + Nja))
        options = np.argwhere(Q == np.max(Q)).flatten()
        return actions[np.random.choice(options)]

    def external_sample_update(self, player, other_player_strategies, sequence_form):
        """
        conducts a more efficient external sample given the opponent strategies
        :param player:
        :param other_player_strategies:
        :param sequence_form:
        :return:
        """

    def observe_utility(self, player, utility, observed_infosets=None, ignore_optimization=False):
        """
        sends utility to player's regret minimizers
        :param player:
        :param utility:
        :return:
        TODO: optimize when lots of zeros
        """
        if player not in self.player_to_regret_minimizers:
            # no infosets of player were ever reached
            # no regret minimizers to observe utility
            return
        tfsdp = self.single_player_trees[player]

        Q = dict()
        # Q[(j,a)] will accumulate u[(j,a)]+sum_{j' in rho'(j,a)} sum_{a'} b_{j'}[a']*Q[(j',a')]
        #  for all actions a, and rho'(j,a) all infosets that are immediate descendants of (j,a)
        #   i.e. in the TFSDP, rho'(j,a) is all descendants j' of j such that there are only signal nodes between them
        #   it is clear that doing this is equivalent to CFR, which sums the V across signal nodes
        # Q collects u[(j,a)] from the one time that j appears in tfsdp
        # Q collects each b_{j'}[a']*Q[(j',a')] since tfsdp is gone through in bottom-up order, and the Q[(j',a')] are computed and added to Q[(j,a)]

        # equivalent to normal CFR, where V[j] = sum_a b_j[a]*Q[(j,a)] for decision nodes, and signal nodes are ignored
        # also this has nicely that Q[(j,a)] = u[(j,a)]+V[rho(j,a)]
        #  this is done to avoid the need for explicit signal nodes
        for infoset_id, dic in reversed(tfsdp.items()):
            if infoset_id not in observed_infosets:
                # all utilities at this infoset are zero
                # and the value of this infoset was never updated by a descendant
                continue
            parent_sequence, legal_actions = dic[PARENT_SEQUENCE], dic[LEGAL_ACTIONS]
            # adds u[(j,a)] to whatever is in Q[(j,a)]
            for a in legal_actions:
                seq = (infoset_id, a)
                Q[seq] = Q.get(seq, 0.) + utility.get((infoset_id, a), 0.)

            # let j',a' be the parent seq of j
            # let V_j = sum_a {b_j[a]Q[(j,a)]}
            #  this adds V_j whatever is in Q[(j',a')]
            #  since tfsdp is in topdown order, no children of j will be reached after this iteration
            #   thus, Q[(j,a)] is at its final value when it is added
            if parent_sequence is not None:
                last_local_strategy = self.player_to_regret_minimizers[player][infoset_id].last_strat
                # only look at actions in support of last local strategy
                V_infoset = sum(last_local_strategy[a]*Q[(infoset_id, a)] for a in last_local_strategy)
                Q[parent_sequence] = Q.get(parent_sequence, 0.) + V_infoset
                # add the parent to infosets with observed utilities
                observed_infosets.add(parent_sequence[0])

        # Q is ONLY nonzero at (j,a) for j in observed_infosets
        # for every line where Q[seq] is updated above, it either is required that
        #  the infoset at seq is in observed_infosets, or the infoset is added immediately after
        if ignore_optimization:
            infosets_to_update = tfsdp.keys()  # update all infosets anyway
        else:
            infosets_to_update = observed_infosets
        for infoset_id in infosets_to_update:
            if infoset_id in tfsdp:
                actions = self.player_to_regret_minimizers[player][infoset_id].action_set
                ute = {a: Q[(infoset_id, a)] for a in actions}
                self.player_to_regret_minimizers[player][infoset_id].observe_utility(utility=ute)

    def compute_player_value(self, player, player_sequential_strategies):
        terminal_seq_utilities, represented_infosets = self.compute_utilities(player=player,
                                                                              other_player_strategies=player_sequential_strategies,
                                                                              sequential_form=True)
        immediate_reward = 0.
        if None in terminal_seq_utilities:
            immediate_reward = terminal_seq_utilities.pop(None)
        sq_strat = player_sequential_strategies[player]

        return immediate_reward + sum(
            sq_strat.get(terminal_seq, 0.)*ute for terminal_seq, ute in terminal_seq_utilities.items())

    def compute_utilities(self, player, other_player_strategies, sequential_form=False):
        """
        :param player: player i that is observing utilities given the current game tree and strategies of other players
        :param other_player_strategies: for each player j!=i, other_player_strategies[j][infoset id of j] is a policy for this infoset
            IS BEHAVIORAL BY DEFAULT
        :return: terminal sequence -> counterfactual utility of player if game terminates before next player action
            includes None as the cf utility of the player if the game ends immediately
        """
        # utilities for player i at terminal sequences
        terminal_sequence_utilities = dict()
        represented_infosets = set()
        if sequential_form:
            # stores node, dictionary of opponent -> opponent reach probability
            frontier = [(self.root, dict())]
        else:
            # stores node, product of all opponent reach probabilities to that node
            frontier = [(self.root, 1.)]

        while frontier:
            node, opponent_reach_prob = frontier.pop()
            if node.terminal:
                parent_seq = node.get_parent_sequence(player)
                utility = node.data[RETURNS][player]
                # multiply utility by the probability that opponents and chance reach it
                # if sequential form and multiplayer, we will need to take the product of opponent reach probabilities
                if sequential_form:
                    temp = 1
                    for _, v in opponent_reach_prob.items():
                        temp = temp*v
                    opponent_reach_prob = temp
                # if behavioral, we directly store the product, so we just need to multiply by the chance reach probability
                external_reach_prob = opponent_reach_prob*node.reach_prob_chance

                # if parent seq is None, this means the game can end before player can act
                # this is added to the value at None
                terminal_sequence_utilities[parent_seq] = external_reach_prob*utility + terminal_sequence_utilities.get(
                    parent_seq, 0.)
                represented_infosets.add(parent_seq if parent_seq is None else parent_seq[0])
                continue
            else:
                for action in node.legal_actions:
                    # compute the opponent reach prob of taking action from node
                    if node.player >= 0 and node.player != player:
                        if sequential_form:
                            child_opponent_reach_prob = opponent_reach_prob.copy()
                            # set this to node.player's strategy at sequence (j,action), as this is reach probability for this opponent
                            # if (node.infoset_id, action) is not in the other player's strategy, we assume it is 0
                            # assert (node.infoset_id, action) in other_player_strategies[node.player]
                            child_opponent_reach_prob[node.player] = other_player_strategies[node.player].get(
                                (node.infoset_id, action), 0.)
                        else:
                            # behavioral, multiply opponent reach prob by probability opponent chooses this
                            if not (node.infoset_id in other_player_strategies[node.player] and
                                    action in other_player_strategies[node.player][node.infoset_id]):
                                # in this case, assume the opponent never plays this action
                                child_opponent_reach_prob = 0
                            else:
                                child_opponent_reach_prob = opponent_reach_prob* \
                                                            other_player_strategies[node.player][node.infoset_id][
                                                                action]
                    else:
                        # chance node or player node, opponent reach prob is unchanged
                        child_opponent_reach_prob = opponent_reach_prob
                    if action in node.children:
                        # we need to recurse on this child with the appropriate opponent reach prob
                        # do not go further if the opponent reach probability is zero
                        if ((sequential_form and all(prob > 0 for _, prob in child_opponent_reach_prob.items())) or
                                (not sequential_form and child_opponent_reach_prob > 0)):
                            frontier.append((node.children[action], child_opponent_reach_prob))
                    else:
                        # this action reaches outside the tree
                        # compute the external reach probability of node
                        if sequential_form:
                            # if sequential, produce the opponent reach probability by doing this product
                            temp = 1.
                            for _, v in child_opponent_reach_prob.items():
                                temp *= v
                            child_opponent_reach_prob = temp
                        # otherwise, the opponent reach probability is already this product, and we multiply by chance reach probability
                        external_reach_prob = child_opponent_reach_prob*node.reach_prob_chance
                        if node.is_chance_node():
                            external_reach_prob = external_reach_prob*node.data[CHANCE_OUTCOMES][action]

                        # compute utility for player at node
                        utility = 0.
                        if EVALUATION in node.data:
                            utility = node.data[EVALUATION][player]
                        elif RETURNS in node.data:
                            utility = node.data[RETURNS][player]

                        if node.player == player:
                            # TODO: WHAT TO DO HERE?
                            #  currently, player recieves the value at node no matter what action they take
                            seq = (node.infoset_id, action)
                            terminal_sequence_utilities[
                                seq] = external_reach_prob*utility + terminal_sequence_utilities.get(seq, 0.)
                            represented_infosets.add(node.infoset_id)
                        else:
                            # player recieves the value at node, weighted by the external reach probability of (NODE, ACTION)
                            # i.e. if node has one action unexpanded, the node's estimated value is weighted by the external reach probability of that action
                            parent_seq = node.get_parent_sequence(player)
                            terminal_sequence_utilities[
                                parent_seq] = external_reach_prob*utility + terminal_sequence_utilities.get(parent_seq,
                                                                                                            0.)
                            represented_infosets.add(parent_seq if parent_seq is None else parent_seq[0])

        return terminal_sequence_utilities, represented_infosets

    def convert_to_sequence_form(self, player, behavioral_strat):
        """
        converts a behavioral strategy to sequential
        requires that beahvioral strat has support over a tree subset of the players tree that includes the root
            i.e. some branches can be cut off, but every infoset that has a parent sequence must have that parent sequence
                available in behavorial strat
                also any infoset in behavioral strat must have been witnessed in tree
        :param player:
        :param behavioral_strat:
        :return:
        """
        if player not in self.single_player_trees:
            return behavioral_strat

        # TODO: dont want to accidently copy the the nodes with .copy(), so we place them in manually
        sq_strat = dict()
        # in topdown order, so sq_strat[parent_seq] is already updated, if not None
        for infoset_id, dic in self.single_player_trees[player].items():
            parent_seq = dic[PARENT_SEQUENCE]
            prob_flow = 1.
            if parent_seq is not None:
                # assert parent_seq in sq_strat, "behavioral strat must be upward closed"
                if parent_seq not in sq_strat:
                    # parent sequence is not observed, so behavioral strat is not upward closeed
                    # equivalent to sq_strat[parent_seq]=0
                    #  also equivalent to not including all descendents of  parent_seq
                    #  thus, we can ignore this

                    continue
                prob_flow = sq_strat[parent_seq]
            if infoset_id in behavioral_strat:
                # if this is not in behavioral strat,
                #  we ignore it and any descendants (which also should not be in behavioral strat)

                # NOTE we are using actions in behavioral_strat[infoset_id] instead of dic[LEGAL_ACTIONS}
                #  if behavioral_strat does not include all actions, we do not include the associated sequences in
                #  the sequence form strat
                for action in behavioral_strat[infoset_id]:
                    # each sequence is updated exactly once, since each infoset appears once in topdown ordering
                    seq = (infoset_id, action)

                    sq_strat[seq] = behavioral_strat[infoset_id][action]*prob_flow

        return sq_strat

    def obtain_strategy(self, player, sequence_form=False):
        """
        obtains strategy for player from regret minimizers
            BEHAVIORAL BY DEFAULT
        :param player:
        :param sequence_form: whether to use sequence form
        :return:
        """
        if player not in self.player_to_regret_minimizers:
            return dict()
        behavioral = {infoset_id: regret_minimizer.next_strategy()
                      for infoset_id, regret_minimizer in self.player_to_regret_minimizers[player].items()}
        if sequence_form:
            return self.convert_to_sequence_form(player=player, behavioral_strat=behavioral)
        else:
            return behavioral

    def constant_sum_nash_gap(self, player_strategies, sequential_form=False, constant_for_constant_sum=0):
        # assumes constant (zero) sum
        gap = 0
        for player in player_strategies:
            gap += self.best_response_value(player=player, other_player_strategies=player_strategies,
                                            sequential_form=sequential_form)
        return gap - constant_for_constant_sum

    def best_response_value(self, player, other_player_strategies, sequential_form=False):
        utilities, rep_infosets = self.compute_utilities(player=player, other_player_strategies=other_player_strategies,
                                                         sequential_form=sequential_form)
        if player in self.single_player_trees:
            for infoset_id, dic in reversed(self.single_player_trees[player].items()):
                parent_seq, legal_actions = dic[PARENT_SEQUENCE], dic[LEGAL_ACTIONS]
                # it is possible that this infoset is never reached in the utility calculation
                if infoset_id not in rep_infosets:  # all utilities will be zero
                    # assert all([utilities.get((infoset_id, a), 0.)==0 for a in legal_actions])
                    continue
                max_ev = max(utilities.get((infoset_id, a), 0.) for a in legal_actions)
                utilities[parent_seq] = utilities.get(parent_seq, 0.) + max_ev
                rep_infosets.add(parent_seq if parent_seq is None else parent_seq[0])
        # None is the root node, which will collect the overall best response value, in addition to the utility
        #  of the game just immediately ending (leftover from compute_utilities)
        return utilities[None]

    def best_response_strategy(self, player, other_player_strategies, sequential_form=False,
                               return_sequential_form=False):
        # TODO: optimize when zeros
        utilities, rep_infosets = self.compute_utilities(player=player, other_player_strategies=other_player_strategies,
                                                         sequential_form=sequential_form)
        strategy = dict()
        for infoset_id, dic in reversed(self.single_player_trees[player].items()):
            parent_seq, legal_actions = dic[PARENT_SEQUENCE], dic[LEGAL_ACTIONS]

            strategy[infoset_id] = {a: 0. for a in legal_actions}
            best_action = max(legal_actions, key=lambda a: utilities.get((infoset_id, a), 0.))
            strategy[infoset_id][best_action] = 1.

            max_ev = utilities.get((infoset_id, best_action), 0.)
            utilities[parent_seq] = utilities.get(parent_seq, 0.) + max_ev
        if return_sequential_form:
            strategy = self.convert_to_sequence_form(player=player, behavioral_strat=strategy)
        return strategy, utilities[None]

    def uniform_strategy(self, player, sequence_form=False):
        """
        produces a uniform strategy for a given player
            BEHAVIORAL BY DEFAULT
        :return dict of (player infoset id -> (aciton -> probability))
        """
        # no available infosets
        if player not in self.player_to_regret_minimizers:
            return dict()

        behavioral = {infoset_id: {a: 1/len(regret_minimizer.action_set) for a in regret_minimizer.action_set}
                      for infoset_id, regret_minimizer in self.player_to_regret_minimizers[player].items()}
        if sequence_form:
            return self.convert_to_sequence_form(player=player, behavioral_strat=behavioral)
        else:
            return behavioral

    def create_full_tree(self):
        """
        debug method, immediately creates full tree
        """
        unexpanded = [self.root]
        while unexpanded:
            node: GameNode = unexpanded.pop()
            if node.terminal:
                continue
            state = self.state_of(node)
            # legal actions always exist for non-terminal nodes
            legal_actions = node.legal_actions
            for action in legal_actions:
                if action not in node.children:
                    leaf = self.make_leaf(node=node,
                                          state=state.clone(),
                                          action=action,
                                          )
                    unexpanded.append(leaf)
                else:
                    unexpanded.append(node.children[action])

    def state_of(self, node):
        state = self.root_state.clone()
        for a in node.get_history():
            state.apply_action(a)
        return state
    def iterate_nodes(self,node=None):
        """
        debug method, iterate over nodes
        """
        if node is None:
            node = self.root
        yield node
        for _, c in node.children.items():
            for n in self.iterate_nodes(c):
                yield n

    def count_nodes(self, node=None):
        s=0
        for _ in self.iterate_nodes(node):
            s+=1
        return s
    def count_infosets(self,node=None):
        all_infosets=[set() for player in self.single_player_trees]
        for node in self.iterate_nodes(node=node):
            if (not node.terminal) and (not node.is_chance_node()):
                all_infosets[node.player].add(node.infoset_id)
        return [len(infosets) for infosets in all_infosets]
    def is_valid_sf_strat(self, player, strategy):
        if player not in self.single_player_trees:
            assert len(strategy) == 0, "this player does not have a decision node yet"
            return True
        for infoset_id, dic in self.single_player_trees[player].items():
            parent_seq = dic[PARENT_SEQUENCE]
            legal_actions = dic[LEGAL_ACTIONS]
            sum_prob = 0
            for action in legal_actions:
                if (infoset_id, action) not in strategy:
                    assert not any((infoset_id, a) in strategy for a in
                                   legal_actions), "only part of the actions in an infoset are represented in policy"
                    sum_prob = None
                    break
                else:
                    sum_prob += strategy[(infoset_id, action)]
            prob_flow = 1.
            if parent_seq is not None:
                prob_flow = strategy[parent_seq]

            assert np.isclose(sum_prob,
                              prob_flow), f"probability sum {sum_prob} is not the parent probabilty flow {prob_flow}"
        for infoset_id, action in strategy:
            assert infoset_id in self.single_player_trees[player], f"infoset not found: {infoset_id}"
            assert action in self.single_player_trees[player][infoset_id][LEGAL_ACTIONS], f"action invalid: {action}"

        return True


def enforce_sq_constraints(single_player_tree, sq_strat, default_prob_flow=1.):
    for infoset_id, dic in single_player_tree.items():
        legal_actions = dic[LEGAL_ACTIONS]
        if any((infoset_id, a) in sq_strat for a in legal_actions):
            parent_seq = dic[PARENT_SEQUENCE]
            prob_flow = default_prob_flow
            if parent_seq is not None:
                prob_flow = sq_strat[parent_seq]
            s = sum(sq_strat.get((infoset_id, a), 0.) for a in legal_actions)

            if not np.isclose(s, prob_flow):
                if s == 0:
                    print(prob_flow)
                    print([sq_strat.get((infoset_id, a), None) for a in legal_actions])
                    raise Exception("HERE")
                c = prob_flow/s
                for a in legal_actions:
                    if (infoset_id, a) in sq_strat:
                        sq_strat[(infoset_id, a)] = sq_strat[(infoset_id, a)]*c


def update_and_produce_avg_strats(sum_sq_0, sum_sq_1,
                                  x0, x1, b0, b1,
                                  accumulated_weight, gtcfr,
                                  i, enforce_sq_form=False):
    # needs behavioral strats for this case:
    #  say infoset j was just added (i.e. it does not exist in sum_sq_0)
    #  then we would like to use x0's values for infoset j, and enforce probability constraints to make this a valid sf strategy
    #  however if x0 is sequence form and happens to have reach prob 0 for infoset j, we have an issue (what should we make this
    #  thus, we will use the behavioral strategy instead in this case
    w = i
    # TODO: sequential trategy averaging needs to be smarter
    #  only need to reweight the infosets that do not appear in sum_sq_0
    for seq in x0:
        if seq in sum_sq_0:
            sum_sq_0[seq] = sum_sq_0[seq] + w*x0[seq]
        else:
            sum_sq_0[seq] = b0[seq[0]][seq[1]]
    for seq in x1:
        if seq in sum_sq_1:
            sum_sq_1[seq] = sum_sq_1[seq] + w*x1[seq]
        else:
            sum_sq_1[seq] = b1[seq[0]][seq[1]]

    for p in [0, 1]:
        sum_sq = [sum_sq_0, sum_sq_1][p]
        if p not in gtcfr.single_player_trees:
            continue
        if enforce_sq_form:
            enforce_sq_constraints(sq_strat=sum_sq,
                                   single_player_tree=gtcfr.single_player_trees[p],
                                   default_prob_flow=accumulated_weight + w)

    avg_sq_0 = {k: v/(accumulated_weight + w) for (k, v) in sum_sq_0.items()}
    avg_sq_1 = {k: v/(accumulated_weight + w) for (k, v) in sum_sq_1.items()}
    return avg_sq_0, avg_sq_1, w


if __name__ == '__main__':
    import time, os
    import matplotlib.pyplot as plt

    game_name = 'leduc_poker'
    game = pyspiel.load_game(game_name)

    gtcfr = GTCFR(
        root_state=PyspielStateStructure(game.new_initial_state()),
        rm_class=PredictiveRegretMatchingPlus,
        #rm_kwargs={'alpha': 1.5}
    )
    do_full = True
    full_gtcfr = GTCFR(
        root_state=PyspielStateStructure(game.new_initial_state()),
        rm_class=DCFRRegretMatching,
        rm_kwargs={'alpha': 1.5}
    )
    full_gtcfr.create_full_tree()
    print('full tree size:', full_gtcfr.count_nodes())
    print('num infosets:', {p: len(rms) for p, rms in full_gtcfr.player_to_regret_minimizers.items()})
    sum_sq_0 = dict()
    sum_sq_1 = dict()
    full_sum_sq_0 = dict()
    full_sum_sq_1 = dict()
    accumulated_weight = 0.
    full_update_times = []
    update_times = []
    extended_nash_gaps = []

    extended_sf_0, extended_sf_1 = dict(), dict()
    for i in range(1, 10000):
        start = time.time()
        # GTCFR update, select and expand a leaf
        expanding_player = i%2

        player_bhv_strategies = {p: gtcfr.obtain_strategy(player=p) for p in [0, 1]}
        node, action = gtcfr.sample_leaf_spot(player_bhv_strategies=player_bhv_strategies,
                                              expanding_players=[expanding_player],
                                              )
        state = gtcfr.state_of(node)
        if not node.terminal:
            leaf = gtcfr.make_leaf(node=node,
                                   state=state,
                                   action=action)
        else:
            leaf = node
        gtcfr.evaluate_and_push(
            players=[expanding_player],
            node=leaf,
            player_strategies=player_bhv_strategies,
            sequential_form=False,
        )
        # regret updates for both full tree and gtcfr
        avg_sq_0, avg_sq_1 = dict(), dict()
        full_avg_sq_0, full_avg_sq_1 = dict(), dict()
        update_times.append(time.time() - start)
        start = time.time()

        w = 0
        for full in [True, False]:
            if not do_full and full:
                continue
            start = time.time()
            tree = full_gtcfr if full else gtcfr

            bhv_0 = tree.obtain_strategy(player=0)
            bhv_1 = tree.obtain_strategy(player=1)
            x0 = tree.convert_to_sequence_form(player=0, behavioral_strat=bhv_0)
            x1 = tree.convert_to_sequence_form(player=1, behavioral_strat=bhv_1)

            u0, obs_infosets0 = tree.compute_utilities(player=0, other_player_strategies={1: bhv_1})
            u1, obs_infosets1 = tree.compute_utilities(player=1, other_player_strategies={0: bhv_0})
            tree.observe_utility(player=0, utility=u0, observed_infosets=obs_infosets0)
            tree.observe_utility(player=1, utility=u1, observed_infosets=obs_infosets1)
            if full:
                full_avg_sq_0, full_avg_sq_1, w = update_and_produce_avg_strats(
                    gtcfr=tree,
                    accumulated_weight=accumulated_weight,
                    sum_sq_0=full_sum_sq_0, sum_sq_1=full_sum_sq_1,
                    x0=x0, x1=x1,
                    b0=bhv_0, b1=bhv_1,
                    i=i, enforce_sq_form=True)
                full_update_times.append(time.time() - start)

            else:
                avg_sq_0, avg_sq_1, w = update_and_produce_avg_strats(
                    gtcfr=tree,
                    accumulated_weight=accumulated_weight,
                    sum_sq_0=sum_sq_0, sum_sq_1=sum_sq_1,
                    x0=x0, x1=x1,
                    b0=bhv_0, b1=bhv_1,
                    i=i, enforce_sq_form=True)
                update_times[-1] += (time.time() - start)

        accumulated_weight += w
        # produce gtcfr strategies for full game by setting the non-visited nodes to uniform
        # i.e. start with uniform sf strategy, update the infosets seen in avg strategies, and enforce prob flow
        extended_sf_0 = full_gtcfr.uniform_strategy(player=0, sequence_form=True)
        extended_sf_0.update(avg_sq_0)
        enforce_sq_constraints(single_player_tree=full_gtcfr.single_player_trees[0],
                               sq_strat=extended_sf_0,
                               default_prob_flow=1.
                               )

        extended_sf_1 = full_gtcfr.uniform_strategy(player=1, sequence_form=True)
        extended_sf_1.update(avg_sq_1)
        enforce_sq_constraints(single_player_tree=full_gtcfr.single_player_trees[1],
                               sq_strat=extended_sf_1,
                               default_prob_flow=1.
                               )

        value0 = gtcfr.compute_player_value(player=0, player_sequential_strategies={0: avg_sq_0, 1: avg_sq_1})
        extended_value0_agnt_uniform = full_gtcfr.compute_player_value(player=0,
                                                                       player_sequential_strategies={
                                                                           0: extended_sf_0,
                                                                           1: full_gtcfr.uniform_strategy(player=1,
                                                                                                          sequence_form=True)
                                                                       })

        # gap = gtcfr.constant_sum_nash_gap(player_strategies={0: avg_sq_0, 1: avg_sq_1}, sequential_form=True)
        extended_gap = full_gtcfr.constant_sum_nash_gap(player_strategies={0: extended_sf_0, 1: extended_sf_1},
                                                        sequential_form=True)
        extended_nash_gaps.append(extended_gap)
        if do_full:
            full_gap = full_gtcfr.constant_sum_nash_gap(player_strategies={0: full_avg_sq_0, 1: full_avg_sq_1},
                                                        sequential_form=True)
        else:
            full_gap = None
        print(i, '(full,gt) nash gaps:', (full_gap, extended_gap),
              '; p0 tree val:', value0,
              '; ext. p0 vs. uniform:', extended_value0_agnt_uniform)
    DIR = os.path.dirname(__file__)
    save_path = os.path.join(DIR, 'output', )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, 'gtcfr_' + game_name + '_conv_by_time')
    gap_over_time = np.array([np.cumsum(update_times),
                              extended_nash_gaps])
    np.save(save_file, gap_over_time)
    plt.plot(gap_over_time[0], gap_over_time[1])
    plt.savefig(os.path.join(save_path, 'gtcfr_' + game_name + '_conv_by_time.png'))
    plt.show()
    print('expanded tree size:', gtcfr.count_nodes())
    print('num infosets:', {p: len(rms) for p, rms in gtcfr.player_to_regret_minimizers.items()})
    print('gtcfr total update time', sum(update_times))
    print('cfr total update time', sum(full_update_times))
    player = 1
    # play against the extended gtcfr strategies (uniform when not in the tree)
    opp_policies = {0: extended_sf_0, 1: extended_sf_1}
    while True:
        s = PyspielStateStructure(game.new_initial_state())
        while not s.is_terminal():
            if s.is_chance_node():
                action = s.sample_chance_outcomes()
            elif s.current_player() == player:
                print(s.state.observation_string())
                for a in s.legal_actions():
                    print(str(a) + ':', s.state.action_to_string(a))
                action = None
                while True:
                    action = input('type action:')
                    if action in [str(a) for a in s.legal_actions()]:
                        break
                    else:
                        print('bad choice, try again')
                for a in s.legal_actions():
                    if str(a) == action:
                        action = a
            else:
                j = s.get_infoset_id()

                policy = {a: opp_policies[s.current_player()][(j, a)] for a in s.legal_actions()}
                p = np.array([prob for _, prob in policy.items()])
                print(p)

                action = np.random.choice([a for a, _ in policy.items()], p=p/np.sum(p))
            s.apply_action(action=action)

        print('your result:', s.returns()[player])
        if input('quit [y/n]: ').lower() == 'y':
            break

    node = gtcfr.root

    s = PyspielStateStructure(game.new_initial_state())
    print(s.state)
    print('infoset:', s.get_infoset_id())
    print(node.player)
    print('chance reach prob:', node.reach_prob_chance)
    print(node.get_history())
    print(node.get_full_player_sequence(0))
    print(node.get_full_player_sequence(1))
    print()
    while not node.terminal:
        a = np.random.choice(list(node.children.keys()))
        node = node.children[a]
        s.apply_action(a)
        print(s.state)
        print('infoset:', s.get_infoset_id())
        print(node.player)
        print('chance reach prob:', node.reach_prob_chance)
        print(node.get_history())
        print(node.get_full_player_sequence(0))
        print(node.get_full_player_sequence(1))
        print(node.data.get(RETURNS, None))
        print()
