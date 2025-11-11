import pyspiel


class Node:
    def __init__(self,
                 parent,
                 player,
                 infoset_id,
                 children=None,
                 last_action=None,
                 terminal=False,
                 **kwargs):
        self.parent = parent
        self.player = player
        self.last_action = last_action
        self.infoset_id = infoset_id
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
        if self.state.current_player() >= 0:
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

                         returns=root_state.returns(),
                         legal_actions=root_state.legal_actions(),
                         )
        self.leaves = {self.root}
        self.regret_minimizers = {}
        self.maybe_add_regret_minimizer(state=root_state)

    def maybe_add_regret_minimizer(self, state: StateStructure):
        if state.current_player() >= 0:
            infoset_id = state.get_infoset_id()
            if infoset_id not in self.regret_minimizers:
                self.regret_minimizers[infoset_id] = self.rm_class(action_set=state.legal_actions(),
                                                                   **self.rm_kwargs)

    def compute_utilities(self, player, other_player_behavioral):
        """
        :param player: player i that is observing utilities given the current game tree and strategies of other players
        :param other_player_behavioral: for each player j!=i, other_player_behavioral[j][infoset id of j] is a policy for this infoset
        :return:
        """

    def make_leaf(self, node, state, action):
        assert action not in node.children
        state_prime = state.child(action)
        leaf = Node(
            parent=node,
            player=state_prime.current_player(),
            last_action=action,
            terminal=state_prime.is_terminal(),
            infoset_id=state_prime.get_infoset_id(),

            returns=state_prime.returns(),
        )
        if not leaf.terminal:
            # add legal actions if not terminal
            leaf.update_data(legal_actions=state_prime.legal_actions())
        if state_prime.is_chance_node():
            leaf.update_data(chance_outcomes=state_prime.chance_outcomes())
        node.children[action] = leaf
        self.maybe_add_regret_minimizer(state=state_prime)

        self.leaves.discard(node)
        self.leaves.add(leaf)
        return leaf

    def count_nodes(self, node=None):
        """
        debug method
        :return: number of nodes in tree including node
        """
        if node is None:
            node = self.root
        return 1 + sum(self.count_nodes(c) for _, c in node.children.items())

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
            legal_actions = node.data['legal_actions']
            if len(node.children) < len(legal_actions):
                state = self.root_state.clone()
                for a in node.get_history():
                    state.apply_action(a)
                for action in legal_actions:
                    if action not in node.children:
                        leaf = self.make_leaf(node=node,
                                              state=state.clone(),
                                              action=action,
                                              )
                        unexpanded.append(leaf)


if __name__ == '__main__':
    game = pyspiel.load_game('tic_tac_toe')

    gtcfr = GTCFR(root_state=PyspielStateStructure(game.new_initial_state()))
    gtcfr.create_full_tree()
    print('full tree size:', gtcfr.count_nodes())
    print('num infosets:', len(gtcfr.regret_minimizers))
    print('num leaves:',len(gtcfr.leaves))

    node = gtcfr.root
    import numpy as np

    s = PyspielStateStructure(game.new_initial_state())
    print(s.state)
    print('infoset:', s.get_infoset_id())
    print(node.player)
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
        print(node.get_history())
        print(node.get_sequence(0))
        print(node.get_sequence(1))
        print(node.data.get('returns'))
        print()
