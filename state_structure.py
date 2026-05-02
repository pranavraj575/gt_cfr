import numpy as np
import pyspiel
import torch


class StateStructure:
    def get_infoset_id(self):
        raise NotImplementedError

    def get_infoset_tensor(self):
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

    def sample_chance_outcomes(self):
        o = self.chance_outcomes()
        return np.random.choice([a for a, _ in o.items()], p=[prob for _, prob in o.items()])

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
    def __init__(self, state: pyspiel.State, use_observation_as_infostate=False):
        self.state = state
        game: pyspiel.Game = self.state.get_game()
        self.use_obs = use_observation_as_infostate
        if self.use_obs:
            self.tensor_shape = game.observation_tensor_shape()
        else:
            self.tensor_shape = game.information_state_tensor_shape()

    def get_infoset_id(self):
        player = self.state.current_player()
        if player >= 0:
            return self.state.information_state_string()
        else:
            return None

    def get_infoset_tensor(self):
        player = self.state.current_player()

        if player >= 0:
            if self.use_obs:
                return torch.tensor(self.state.observation_tensor()).reshape(self.tensor_shape).unsqueeze(0)
            else:
                return torch.tensor(self.state.information_state_tensor()).reshape(self.tensor_shape).unsqueeze(0)
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
                if state.is_chance_node():
                    action = state.sample_chance_outcomes()
                else:
                    actions = list(state.legal_actions())
                    action = np.random.choice(actions)
                state.apply_action(action)
            returns += np.array(state.returns())
        return returns / n


if __name__ == "__main__":
    import os
    import ast
    from config_networks import CustomNN

    g: pyspiel.Game = pyspiel.load_game("tic_tac_toe")
    s = PyspielStateStructure(state=g.new_initial_state(), use_observation_as_infostate=True)
    s.apply_action(2)
    s.apply_action(3)

    print(s.get_infoset_tensor().shape)
    config_dir = os.path.join(os.path.dirname(os.path.basename(__file__)), "config_files")
    f = open(os.path.join(config_dir, "ttt_net.txt"), "r")
    ttt_net = CustomNN(structure=ast.literal_eval(f.read()))
    f.close()
    print(ttt_net)
    print(ttt_net(s.get_infoset_tensor()))
    optim = torch.optim.Adam(ttt_net.parameters())
    for _ in range(100):
        optim.zero_grad()
        loss = torch.mean(torch.square(ttt_net(s.get_infoset_tensor())))
        loss.backward()
        optim.step()
        print(loss.item())
