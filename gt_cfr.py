import pyspiel
import numpy as np


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

    def get_sequence(self, player):
        if self.parent is None:
            return ()

        if self.parent.player == player:
            return self.parent.get_sequence(player) + (self.last_action,)
        else:
            return self.parent.get_sequence(player)


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
            return self.state.observation_string()
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


class RegretMatching(object):
    def __init__(self, action_set, **kwargs):
        self.action_set = set(action_set)
        self.cum_regrets = {a: 0. for a in self.action_set}
        self.last_strat = None

    def next_strategy(self):
        # You might want to return a dictionary mapping each action in
        # `self.action_set` to the probability of picking that action
        sum_regrets = sum(max(self.cum_regrets[a], 0) for a in self.action_set)
        if sum_regrets == 0:
            self.last_strat = {
                a: 1/len(self.action_set)
                for a in self.action_set
            }
        else:
            self.last_strat = {
                a: max(self.cum_regrets[a], 0)/sum_regrets
                for a in self.action_set
            }

        return self.last_strat.copy()

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set

        # <x,u>
        ute = sum(self.last_strat[a]*utility[a] for a in self.action_set)

        # r^t = r^{t-1} + (u-<x,u>1)
        self.cum_regrets = {
            # instant regret is u-<x,u>1, at dimension a this is u[a]-<x,u>
            a: self.cum_regrets[a] + (utility[a] - ute)
            for a in self.action_set
        }


class GTCFR:
    def __init__(self, root_state: StateStructure, rm_class=RegretMatching, rm_kwargs=None):
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
                         )
        if self.root.is_chance_node():
            self.root.update_data(chance_outcomes=self.root_state.chance_outcomes())
        self.top_down_order = [self.root]
        # dict of player -> tfsdp
        # tfsdp is (infoset id) -> list of (node, parent sequence) in topdown order
        self.tfsdps = dict()
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
                self.player_to_regret_minimizers[player][infoset_id] = self.rm_class(action_set=state.legal_actions(),
                                                                                     **self.rm_kwargs)

    def sample_leaf_spot(self, player_bhv_strategies):
        """
        samples a node, action that is outside of the tree, using the strategies given
            ensures that either node is terminal (in which case action is None)
            or node,action leads outside the current tree
        :param player_bhv_strategies: for each player i, other_player_strategies[i][infoset id of i] is a policy for this infoset
            IS BEHAVIORAL,
        :return: (node,action)
        """
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
                dist = [strat[a] for a in actions]
            action = np.random.choice(actions, p=dist)
            if action not in node.children:
                # this is outside the tree
                break
            else:
                # recurse on child
                node = node.children[action]
        return (node, action)

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

        for node, parent_sequence in tfsdp[::-1]:
            # adds u[(j,a)] to whatever is in Q[(j,a)]
            for a in node.legal_actions:
                seq = (node.infoset_id, a)
                Q[seq] = Q.get(seq, 0.) + utility.get((node.infoset_id, a), 0.)

            # let j',a' be the parent seq of j
            # let V_j = sum_a {b_j[a]Q[(j,a)]}
            #  this adds V_j whatever is in Q[(j',a')]
            #  since tfsdp is in topdown order, no children of j will be reached after this iteration
            #   thus, Q[(j,a)] is at its final value when it is added
            if parent_sequence is not None:
                last_local_strategy = self.player_to_regret_minimizers[player][node.infoset_id].last_strat
                V_infoset = sum(last_local_strategy[a]*Q[(node.infoset_id, a)] for a in node.legal_actions)
                Q[parent_sequence] = Q.get(parent_sequence, 0.) + V_infoset

        for node, _ in tfsdp:
            j=node.infoset_id
            ute = {a:Q[(j,a)] for a in node.legal_actions}
            self.player_to_regret_minimizers[player][j].observe_utility(utility=ute)

    def compute_utilities(self, player, other_player_bhv_strategies):
        """
        :param player: player i that is observing utilities given the current game tree and strategies of other players
        :param other_player_bhv_strategies: for each player j!=i, other_player_strategies[j][infoset id of j] is a policy for this infoset
            IS BEHAVIORAL
        :return:
        """
        # utilities for player i at terminal sequences
        terminal_sequence_utilities = dict()
        frontier = [(self.root, 1.)]
        while frontier:
            node, opponent_reach_prob = frontier.pop()
            if node.terminal:
                parent_seq = node.get_parent_sequence(player)
                # if parent seq is None, there is no player decision that leads to this leaf, so we can ignore
                if parent_seq is not None:
                    utility = node.data['returns'][player] if 'returns' in node.data else 0.
                    # multiply utility by the probability that opponents and chance reach it
                    external_reach_prob = opponent_reach_prob*node.reach_prob_chance
                    terminal_sequence_utilities[parent_seq] = external_reach_prob*utility + terminal_sequence_utilities.get(parent_seq, 0.)
            else:
                for action in node.legal_actions:
                    # compute the opponent reach prob of taking action from node
                    if node.player >= 0 and node.player != player:
                        # behavioral, multiply opponent reach prob by probability opponent chooses this
                        child_opponent_reach_prob = opponent_reach_prob*other_player_bhv_strategies[node.player][node.infoset_id][action]
                    else:
                        # chance node or player node, opponent reach prob is unchanged
                        child_opponent_reach_prob = opponent_reach_prob

                    if action in node.children:
                        # we need to recurse on this child with the appropriate opponent reach prob
                        frontier.append((node.children[action], child_opponent_reach_prob))
                    else:
                        # this action reaches outside the tree

                        # compute the external reach probability of node
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
        if not leaf.terminal:
            # add legal actions if not terminal
            leaf.update_data(legal_actions=state_prime.legal_actions())
        if leaf.is_chance_node():
            leaf.update_data(chance_outcomes=state_prime.chance_outcomes())
        node.children[action] = leaf
        self.maybe_add_regret_minimizer(state=state_prime)
        self.top_down_order.append(leaf)

        if not leaf.is_chance_node():
            if leaf.player not in self.tfsdps:
                self.tfsdps[leaf.player] = []
            self.tfsdps[leaf.player].append((leaf, leaf.get_parent_sequence()))
        return leaf

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

    def convert_to_sequence_form(self, player, behavioral_strat, inplace=False):
        if player not in self.tfsdps:
            return behavioral_strat

        # TODO: dont want to accidently copy the the nodes with .copy(), so we place them in manually
        sq_strat = behavioral_strat if inplace else {k: v for k, v in behavioral_strat.items()}
        # in topdown order, so sq_strat[parent_seq] is already updated, if not None
        for (node, parent_seq) in self.tfsdps[player]:
            prob_flow = 1.
            if parent_seq is not None:
                prob_flow = sq_strat[parent_seq]
            for action in node.legal_actions:
                # each sequence is updated exactly once, since each infoset appears once in topdown ordering
                seq = (node.infoset_id, action)
                sq_strat[seq] = sq_strat[seq]*prob_flow

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
            return self._cvt_to_sequence_form(player=player, behavioral_strat=behavioral)
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
            # legal actions always exist for non-terminal nodes
            legal_actions = node.legal_actions
            for action in legal_actions:
                if action not in node.children:
                    state = self.root_state.clone()
                    for a in node.get_history():
                        state.apply_action(a)
                    leaf = self.make_leaf(node=node,
                                          state=state.clone(),
                                          action=action,
                                          )
                    unexpanded.append(leaf)
                else:
                    unexpanded.append(node.children[action])


if __name__ == '__main__':
    game = pyspiel.load_game('kuhn_poker')

    gtcfr = GTCFR(root_state=PyspielStateStructure(game.new_initial_state()))
    for _ in range(100):
        node, action = gtcfr.sample_leaf_spot(player_bhv_strategies={i: gtcfr.obtain_strategy(player=i) for i in [0, 1]})
        state = gtcfr.state_of(node)
        if not node.terminal:
            gtcfr.make_leaf(node=node,
                            state=state,
                            action=action)

    print('full tree size:', gtcfr.count_nodes())
    print('num infosets:', {p: len(rms) for p, rms in gtcfr.player_to_regret_minimizers.items()})
    for node in gtcfr.top_down_order:
        print(node.infoset_id, node.get_history())
    gtcfr.create_full_tree()
    print('full tree size:', gtcfr.count_nodes())
    print('num infosets:', {p: len(rms) for p, rms in gtcfr.player_to_regret_minimizers.items()})

    node = gtcfr.root
    import numpy as np

    s = PyspielStateStructure(game.new_initial_state())
    print(s.state)
    print('infoset:', s.get_infoset_id())
    print(node.player)
    print('chance reach prob:', node.reach_prob_chance)
    print(node.get_history())
    print(node.get_sequence(0))
    print(node.get_sequence(1))
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
        print(node.get_sequence(0))
        print(node.get_sequence(1))
        print(node.data.get('returns'))
        print()
    for i in range(100):
        x0 = gtcfr.obtain_strategy(player=0)
        x1 = gtcfr.obtain_strategy(player=1)
        u0 = gtcfr.compute_utilities(player=0, other_player_bhv_strategies={1: x1})
        gtcfr.observe_utility(player=0, utility=u0)
        u1 = gtcfr.compute_utilities(player=1, other_player_bhv_strategies={0: x0})
        gtcfr.observe_utility(player=1, utility=u1)
        print(x0)
