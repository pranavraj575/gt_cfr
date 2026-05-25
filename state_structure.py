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


class DebugStateStructure(StateStructure):
    """
    wrapper to add debug 'bad' actions that should never be expanded by both players
    """

    def __init__(self, state_struct: StateStructure, negative_utlility=-10, info_dict=None):
        self.state_struct = state_struct
        self.negative_utility = negative_utlility
        if info_dict is None:
            info_dict = dict()
        self.info_dict = info_dict

    def get_infoset_id(self):

        player = self.current_player()
        if player >= 0:
            return str(self.state_struct.get_infoset_id()) + "_" + str(self.info_dict.get("bad_action_chosen", "no_bad"))
        else:
            return None

    def get_infoset_tensor(self):
        return self.state_struct.get_infoset_tensor()

    def clone(self):
        return DebugStateStructure(
            state_struct=self.state_struct.clone(),
            negative_utlility=self.negative_utility,
            info_dict=self.info_dict.copy(),
        )

    def current_player(self):
        return self.state_struct.current_player()

    def is_terminal(self):
        return self.state_struct.is_terminal()

    def is_chance_node(self):
        return self.state_struct.is_chance_node()

    def chance_outcomes(self):
        return self.state_struct.chance_outcomes()

    def returns(self):
        returns = self.state_struct.returns()
        if "bad_action_chosen" in self.info_dict:
            utilities = np.ones_like(returns) * (-self.negative_utility / (max(1, len(returns) - 1)))
            utilities[self.info_dict["bad_action_chosen"]] = self.negative_utility
            return utilities
        else:
            return returns

    def apply_action(self, action):
        if action == -1:
            self.info_dict["bad_action_chosen"] = self.current_player()
        else:
            self.state_struct.apply_action(action)

    def child(self, action):
        clone = self.clone()
        clone.apply_action(action)
        return clone

    def legal_actions(self):
        if "bad_action_chosen" in self.info_dict or self.is_chance_node():
            return self.state_struct.legal_actions()
        else:
            return [-1] + list(self.state_struct.legal_actions())

    def evaluate(self):
        return self.state_struct.evaluate()


if __name__ == "__main__":
    import os
    import ast

    s = DebugStateStructure(
        state_struct=PyspielStateStructure(
            state=pyspiel.load_game("kuhn_poker").new_initial_state(),
        ),
        negative_utlility=-100,
    )
    print(s.current_player(), s.legal_actions())

    while s.current_player() != 0:
        s.apply_action(s.legal_actions()[-1])
    print(s.current_player(), s.legal_actions())
    s.apply_action(-1)
    print(s.current_player(), s.legal_actions())
    all_returns = []
    for _ in range(1000):
        sp = s.clone()
        while not sp.is_terminal():
            sp.apply_action(np.random.choice(sp.legal_actions()))
        all_returns.append(sp.returns())
    sorted_returns = np.array([sorted(sr) for sr in all_returns])
    print(np.mean(sorted_returns, axis=0))
    print(np.mean(np.array(all_returns), axis=0))

    g: pyspiel.Game = pyspiel.load_game("tic_tac_toe")
    s = PyspielStateStructure(state=g.new_initial_state(), use_observation_as_infostate=True)
    s.apply_action(2)
    s.apply_action(3)
    from config_networks import CustomNN

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
