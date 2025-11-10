import pyspiel


class Node:
    def __init__(self,
                 parent,
                 player,
                 children=None,
                 last_action=None,
                 terminal=False,
                 **kwargs):
        self.parent = parent
        self.player = player
        self.last_action = last_action
        self.children = children
        if self.children is None:
            self.children = dict()
        self.terminal = terminal
        self.data = kwargs

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


class GTCFR:
    def __init__(self, game: pyspiel.Game):
        self.game = game
        root_state = game.new_initial_state()
        self.root = Node(parent=None,
                         player=root_state.current_player(),
                         terminal=root_state.is_terminal(),

                         returns=root_state.returns(),
                         )

    def make_leaf(self, node, state, action):
        assert action not in node.children
        state_prime = state.child(action)
        leaf = Node(
            parent=node,
            player=state_prime.current_player(),
            last_action=action,
            terminal=state_prime.is_terminal(),

            returns=state_prime.returns(),
        )
        node.children[action] = leaf
        return leaf


if __name__ == '__main__':
    game = pyspiel.load_game('tic_tac_toe')
    gtcfr = GTCFR(game=game)
    unexpanded_leaves = [gtcfr.root]
    count = 1
    while unexpanded_leaves:
        s = game.new_initial_state()
        node: Node = unexpanded_leaves.pop()
        if node.terminal:
            continue
        for a in node.get_history():
            s.apply_action(a)
        actions = s.legal_actions()
        if len(actions) == len(node.children):
            # fully expanded
            continue
        if len(node.children) + 1 < len(actions):
            # even adding a child will leave this node unexpanded
            unexpanded_leaves.append(node)

        action = list(set(actions).difference(node.children.keys()))[0]
        leaf = gtcfr.make_leaf(node=node,
                               state=s,
                               action=action,
                               )
        count += 1
        unexpanded_leaves.append(leaf)
    node = gtcfr.root
    import numpy as np

    s = game.new_initial_state()
    print(s)
    print(node.player)
    print(node.get_history())
    print(node.get_sequence(0))
    print(node.get_sequence(1))
    print()
    while not node.terminal:
        a=np.random.choice(list(node.children.keys()))
        node=node.children[a]
        s.apply_action(a)
        print(s)
        print(node.player)
        print(node.get_history())
        print(node.get_sequence(0))
        print(node.get_sequence(1))
        print(node.data.get('returns'))
        print()
