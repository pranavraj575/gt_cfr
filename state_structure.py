import numpy as np
import pyspiel


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

    def evaluate(self):
        """
        produces an evaluation, utility for each player
        """
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

    def evaluate(self):
        """
        produces an evaluation, utility for each player
        """
        returns = 0
        n = 3
        for _ in range(n):
            state = self.clone()
            while not state.is_terminal():
                actions = list(state.legal_actions())
                if state.is_chance_node():
                    o = state.chance_outcomes()
                    action = np.random.choice(actions, p=[o[a] for a in actions])
                else:
                    action = np.random.choice(actions)
                state.apply_action(action)
            returns += np.array(state.returns())
        return returns/n
