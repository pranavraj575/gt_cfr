import pyspiel
import numpy as np
from collections import OrderedDict

from regret_minimizer import *

STORE_HISTORY = True


class Node:
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
        if 'player_sequences' in self.data and player in self.data['player_sequences']:
            return self.data['player_sequences'][player]
        if self.parent is None:
            return ()
        if self.parent.player == player:
            return self.parent.get_full_player_sequence(player) + ((self.parent.infoset_id, self.last_action),)
        else:
            return self.parent.get_full_player_sequence(player)


class StateStructure():

    def get_infoset_id(self):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    def current_player(self):
        raise NotImplementedError

    def is_terminal(self):
        raise NotImplementedError

    def is_chance_node(self):
        raise NotImplementedError

    def chance_outcomes(self):
        raise NotImplementedError

    def returns(self):
        raise NotImplementedError

    def apply_action(self, action):
        raise NotImplementedError

    def child(self, action):
        clone = self.clone()
        clone.apply_action(action)
        return clone

    def legal_actions(self):
        raise NotImplementedError


class PyspielStateStructure(StateStructure):
    def __init__(self, state: pyspiel.State):
        self.state = state

    def get_infoset_id(self):
        player = self.state.current_player()
        if player >= 0:
            return self.state.information_state_string()
        else:
            return None

    def clone(self):
        return PyspielStateStructure(state=self.state.clone())

    def current_player(self):
        return self.state.current_player()

    def is_terminal(self):
        return self.state.is_terminal()

    def is_chance_node(self):
        return self.state.is_chance_node()

    def chance_outcomes(self):
        return {a: prob for a, prob in self.state.chance_outcomes()}
        return self.state.chance_outcomes()

    def returns(self):
        return self.state.returns()

    def apply_action(self, action):
        self.state.apply_action(action)

    def child(self, action):
        return PyspielStateStructure(state=self.state.child(action))

    def legal_actions(self):
        return self.state.legal_actions()


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

        self.root = Node(parent=None,
                         player=root_state.current_player(),
                         terminal=root_state.is_terminal(),
                         infoset_id=root_state.get_infoset_id(),
                         reach_prob_chance=1.,
                         legal_actions=root_state.legal_actions(),

                         returns=root_state.returns(),
                         history=(),
                         )
        if self.root.is_chance_node():
            self.root.update_data(chance_outcomes=self.root_state.chance_outcomes())

        # dict of player -> tfsdp
        # tfsdp is an ORDERED DICT (infoset id) -> (parent sequence, legal actions, all nodes in infoset)
        #  in topdown order
        self.tfsdps = dict()
        self.add_to_extra_structrues(self.root)

        # dict of player -> (infoset id -> regret minimizer)
        self.player_to_regret_minimizers = dict()
        self.maybe_add_regret_minimizer(state=root_state)

    def maybe_add_regret_minimizer(self, state: StateStructure):
        player = state.current_player()
        if player >= 0:
            if player not in self.player_to_regret_minimizers:
                self.player_to_regret_minimizers[player] = dict()

            infoset_id = state.get_infoset_id()
            if infoset_id not in self.player_to_regret_minimizers[player]:
                rm = self.rm_class(action_set=state.legal_actions(),
                                   **self.rm_kwargs)
                self.player_to_regret_minimizers[player][infoset_id] = rm
                return rm

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
                dist = [node.data['chance_outcomes'][a] for a in actions]
            else:
                strat = player_bhv_strategies[node.player][node.infoset_id]
                dist = np.array([strat[a] for a in actions])
                if node.player in expanding_players:
                    # distribution should be (1-p)*{dist proportional to 1 for support of player strategy} + p*{1 for max of PUCT selection}

                    action = self.PUCT_selection(node)
                    action_idx = actions.index(action)

                    support = (dist > 0)
                    dist = (1 - p)*support/np.sum(support)  # distribution where the player strategy is positive, sums to (1-p)
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

    def external_sample_update(self, player, other_player_strategies, sequence_form):
        """
        conducts a more efficient external sample given the opponent strategies
        :param player:
        :param other_player_strategies:
        :param sequence_form:
        :return:
        """

    def observe_utility(self, player, utility):
        """
        sends utility to player's regret minimizers
        :param player:
        :param utility:
        :return:
        """
        if player not in self.player_to_regret_minimizers:
            # no infosets of player were ever reached
            # no regret minimizers to observe utility
            return
        tfsdp = self.tfsdps[player]

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
        for infoset_id, (parent_sequence, legal_actions, _) in reversed(tfsdp.items()):
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
                V_infoset = sum(last_local_strategy[a]*Q[(infoset_id, a)] for a in legal_actions)
                Q[parent_sequence] = Q.get(parent_sequence, 0.) + V_infoset

        for infoset_id, (_, legal_actions, _) in tfsdp.items():
            ute = {a: Q[(infoset_id, a)] for a in legal_actions}
            self.player_to_regret_minimizers[player][infoset_id].observe_utility(utility=ute)

    def compute_player_value(self, player, player_sequential_strategies):
        terminal_seq_utilities = self.compute_utilities(player=player,
                                                        other_player_strategies=player_sequential_strategies,
                                                        sequential_form=True)
        sq_strat = player_sequential_strategies[player]
        return sum(sq_strat[terminal_seq]*ute for terminal_seq, ute in terminal_seq_utilities.items())

    def compute_utilities(self, player, other_player_strategies, sequential_form=False):
        """
        :param player: player i that is observing utilities given the current game tree and strategies of other players
        :param other_player_strategies: for each player j!=i, other_player_strategies[j][infoset id of j] is a policy for this infoset
            IS BEHAVIORAL BY DEFAULT
        :return:
        """
        # utilities for player i at terminal sequences
        terminal_sequence_utilities = dict()
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
                # if parent seq is None, there is no player decision that leads to this leaf, so we can ignore
                if parent_seq is not None:
                    utility = node.data['returns'][player] if 'returns' in node.data else 0.
                    # multiply utility by the probability that opponents and chance reach it
                    # if sequential form and multiplayer, we will need to take the product of opponent reach probabilities
                    if sequential_form:
                        temp = 1
                        for _, v in opponent_reach_prob.items():
                            temp = temp*v
                        opponent_reach_prob = temp
                    # if behavioral, we directly store the product, so we just need to multiply by the chance reach probability
                    external_reach_prob = opponent_reach_prob*node.reach_prob_chance
                    terminal_sequence_utilities[parent_seq] = external_reach_prob*utility + terminal_sequence_utilities.get(parent_seq, 0.)
                continue
            else:
                for action in node.legal_actions:
                    # compute the opponent reach prob of taking action from node
                    if node.player >= 0 and node.player != player:
                        if sequential_form:
                            child_opponent_reach_prob = opponent_reach_prob.copy()
                            # set this to node.player's strategy at sequence (j,action), as this is reach probability for this opponent
                            child_opponent_reach_prob[node.player] = other_player_strategies[node.player][(node.infoset_id, action)]
                        else:
                            # behavioral, multiply opponent reach prob by probability opponent chooses this
                            child_opponent_reach_prob = opponent_reach_prob*other_player_strategies[node.player][node.infoset_id][action]
                    else:
                        # chance node or player node, opponent reach prob is unchanged
                        child_opponent_reach_prob = opponent_reach_prob

                    if action in node.children:
                        # we need to recurse on this child with the appropriate opponent reach prob
                        frontier.append((node.children[action], child_opponent_reach_prob))
                    else:
                        # this action reaches outside the tree
                        # compute the external reach probability of node
                        if sequential_form:
                            # if sequuential, produce the opponent reach probability by doing this product
                            temp = 1.
                            for _, v in child_opponent_reach_prob.items():
                                temp *= v
                            child_opponent_reach_prob = temp
                        # otherwise, the opponent reach probability is already this product, and we multiply by chance reach probability
                        external_reach_prob = child_opponent_reach_prob*node.reach_prob_chance
                        if node.is_chance_node():
                            external_reach_prob = external_reach_prob*node.data['chance_outcomes'][action]

                        # compute utility for player at node
                        utility = node.data['returns'][player] if 'returns' in node.data else 0.
                        if node.player == player:
                            # TODO: WHAT TO DO HERE?
                            #  currently, player recieves the value at node no matter what action they take
                            seq = (node.infoset_id, action)
                            terminal_sequence_utilities[seq] = external_reach_prob*utility + terminal_sequence_utilities.get(seq, 0.)
                        else:
                            # player recieves the value at node, weighted by the external reach probability of (NODE, ACTION)
                            # i.e. if node has one action unexpanded, the node's estimated value is weighted by the external reach probability of that action
                            parent_seq = node.get_parent_sequence(player)
                            if parent_seq is not None:
                                terminal_sequence_utilities[parent_seq] = external_reach_prob*utility + terminal_sequence_utilities.get(parent_seq, 0.)
        return terminal_sequence_utilities

    def make_leaf(self, node, state, action):
        assert action not in node.children
        state_prime = state.child(action)
        action_prob_chance = 1.
        if node.is_chance_node():
            action_prob_chance = node.data['chance_outcomes'][action]
        leaf = Node(
            parent=node,
            player=state_prime.current_player(),
            last_action=action,
            terminal=state_prime.is_terminal(),
            infoset_id=state_prime.get_infoset_id(),
            reach_prob_chance=node.reach_prob_chance*action_prob_chance,
            legal_actions=state_prime.legal_actions(),

            returns=state_prime.returns(),
        )
        child_player_sequences = node.data.get('player_sequences', dict())
        if not node.is_chance_node():
            child_player_sequences = child_player_sequences.copy() # can just send the same object to chance nodes
            child_player_sequences[node.player] = child_player_sequences.get(node.player, ()) + ((node.infoset_id, action),)
        leaf.update_data(player_sequences=child_player_sequences)

        if leaf.is_chance_node():
            leaf.update_data(chance_outcomes=state_prime.chance_outcomes())
        node.children[action] = leaf
        self.maybe_add_regret_minimizer(state=state_prime)
        self.add_to_extra_structrues(leaf)
        return leaf

    def add_to_extra_structrues(self, node):
        """
        adds to extra structures, including tfsdp for player (if applicable)
        :param node:
        :return:
        """
        if not node.is_chance_node() and not node.terminal:
            # add to tfsdp structure
            if node.player not in self.tfsdps:
                self.tfsdps[node.player] = OrderedDict()
            # reasigning a value does not change order, an infoset will be ordered based on the FIRST time it is seen
            # though doing this just to be safe
            if node.infoset_id not in self.tfsdps[node.player]:
                self.tfsdps[node.player][node.infoset_id] = (node.get_parent_sequence(), node.legal_actions, set())
            _, _, infoset = self.tfsdps[node.player][node.infoset_id]
            infoset.add(node)

    def state_of(self, node):
        state = self.root_state.clone()
        for a in node.get_history():
            state.apply_action(a)
        return state

    def count_nodes(self, node=None):
        """
        debug method
        :return: number of nodes in tree including node
        """
        if node is None:
            node = self.root
        return 1 + sum(self.count_nodes(c) for _, c in node.children.items())

    def convert_to_sequence_form(self, player, behavioral_strat):
        if player not in self.tfsdps:
            return behavioral_strat

        # TODO: dont want to accidently copy the the nodes with .copy(), so we place them in manually
        sq_strat = dict()
        # in topdown order, so sq_strat[parent_seq] is already updated, if not None
        for infoset_id, (parent_seq, legal_actions, _) in self.tfsdps[player].items():
            prob_flow = 1.
            if parent_seq is not None:
                prob_flow = sq_strat[parent_seq]
            for action in legal_actions:
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
            gap += self.best_response_value(player=player, other_player_strategies=player_strategies, sequential_form=sequential_form)
        return gap - constant_for_constant_sum

    def best_response_value(self, player, other_player_strategies, sequential_form=False):
        utilities = self.compute_utilities(player=player, other_player_strategies=other_player_strategies, sequential_form=sequential_form)
        for infoset_id, (parent_sequence, legal_actions, _) in reversed(self.tfsdps[player].items()):
            max_ev = max(utilities[(infoset_id, a)] for a in legal_actions)
            utilities[parent_sequence] = utilities.get(parent_sequence, 0.) + max_ev
        # None is the root node, which will collect the overall best response value
        return utilities[None]

    def best_response_strategy(self, player, other_player_strategies, sequential_form=False, return_sequential_form=False):
        utilities = self.compute_utilities(player=player, other_player_strategies=other_player_strategies, sequential_form=sequential_form)
        strategy = dict()
        for infoset_id, (parent_sequence, legal_actions, _) in reversed(self.tfsdps[player].items()):
            strategy[infoset_id] = {a: 0. for a in legal_actions}
            best_action = max(legal_actions, key=lambda a: utilities[(infoset_id, a)])
            strategy[infoset_id][best_action] = 1.

            max_ev = utilities[(infoset_id, best_action)]

            utilities[parent_sequence] = utilities.get(parent_sequence, 0.) + max_ev
        if return_sequential_form:
            strategy = self.convert_to_sequence_form(player=player, behavioral_strat=strategy)
        return strategy

    def uniform_behavioral_strategy(self, player, sequence_form=False):
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
            node: Node = unexpanded.pop()
            if node.terminal:
                continue

            state = self.root_state.clone()
            for a in node.get_history():
                state.apply_action(a)
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


if __name__ == '__main__':
    game = pyspiel.load_game('tic_tac_toe')

    gtcfr = GTCFR(
        root_state=PyspielStateStructure(game.new_initial_state()),
        rm_class=PredictiveRegretMatchingPlus,
    )
    for _ in range(69):
        node, action = gtcfr.sample_leaf_spot(player_bhv_strategies={i: gtcfr.obtain_strategy(player=i) for i in [0, 1]},
                                              expanding_players=None,
                                              )
        state = gtcfr.state_of(node)
        if not node.terminal:
            gtcfr.make_leaf(node=node,
                            state=state,
                            action=action)

    print('full tree size:', gtcfr.count_nodes())
    print('num infosets:', {p: len(rms) for p, rms in gtcfr.player_to_regret_minimizers.items()})
    gtcfr.create_full_tree()
    if False:
        for player,tfsdp in gtcfr.tfsdps.items():
            print()
            print(player)
            for infoset_id,(parent_seq,legal_actions,infoset) in tfsdp.items():
                print('infoset:',infoset_id,'; actions:',legal_actions,'; size:',len(infoset))
    print('full tree size:', gtcfr.count_nodes())
    print('num infosets:', {p: len(rms) for p, rms in gtcfr.player_to_regret_minimizers.items()})
    br_value = gtcfr.best_response_value(player=0, other_player_strategies={1: gtcfr.uniform_behavioral_strategy(player=1)})
    print('p0 best response value against uniform:', br_value)

    sum_sq_0 = dict()
    sum_sq_1 = dict()
    accumulated_weight = 0.
    for i in range(100):
        bhv_0 = gtcfr.obtain_strategy(player=0)
        bhv_1 = gtcfr.obtain_strategy(player=1)
        # bhv_1 = gtcfr.uniform_behavioral_strategy(player=1)
        u0 = gtcfr.compute_utilities(player=0, other_player_strategies={1: bhv_1})

        gtcfr.observe_utility(player=0, utility=u0)
        u1 = gtcfr.compute_utilities(player=1, other_player_strategies={0: bhv_0})
        gtcfr.observe_utility(player=1, utility=u1)
        x0 = gtcfr.convert_to_sequence_form(player=0, behavioral_strat=bhv_0)
        x1 = gtcfr.convert_to_sequence_form(player=1, behavioral_strat=bhv_1)
        for seq in x0:
            sum_sq_0[seq] = sum_sq_0.get(seq, 0.) + x0[seq]
        for seq in x1:
            sum_sq_1[seq] = sum_sq_1.get(seq, 0.) + x1[seq]
        accumulated_weight += 1
        avg_sq_0 = {k: v/accumulated_weight for (k, v) in sum_sq_0.items()}
        avg_sq_1 = {k: v/accumulated_weight for (k, v) in sum_sq_1.items()}
        value0 = gtcfr.compute_player_value(player=0, player_sequential_strategies={0: avg_sq_0, 1: avg_sq_1})
        gap = gtcfr.constant_sum_nash_gap(player_strategies={0: avg_sq_0, 1: avg_sq_1}, sequential_form=True)
        print(i, 'nash gap', gap, 'p0 value', value0)

    player = 0
    opp_policies = {0: sum_sq_0, 1: sum_sq_1}
    opp_policies = {p: {k: v/accumulated_weight for (k, v) in policy.items()}
                    for p, policy in opp_policies.items()
                    }
    while True:
        s = PyspielStateStructure(game.new_initial_state())
        while not s.is_terminal():
            if s.is_chance_node():
                o = s.chance_outcomes()
                action = np.random.choice([a for a, _ in o.items()], p=[prob for _, prob in o.items()])
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

                action = np.random.choice([a for a, _ in policy.items()], p=p/np.sum(p))
            s.apply_action(action=action)

        print('your result:', s.returns()[player])
        if input('quit [y/n]: ').lower() == 'y':
            break

    node = gtcfr.root
    import numpy as np

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
        print(node.data.get('returns'))
        print()
