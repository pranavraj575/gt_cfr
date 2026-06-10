import torch


class RegretMinimizer(object):
    action_set = set()
    action_list = list()

    def next_strategy(self):
        raise NotImplementedError

    def last_strategy(self):
        raise NotImplementedError

    def observe_utility(self, utility):
        raise NotImplementedError

    def reset(self, warm_start=None):
        raise NotImplementedError


class RegretMatching(RegretMinimizer):
    def __init__(self, action_set, **kwargs):
        self.action_set = action_set
        self.action_list = list(action_set)
        self.n = len(self.action_list)
        self.cum_regrets = torch.zeros(self.n)
        self.last_strat = torch.ones(self.n) / self.n
        self.reset()

    def last_strategy(self):
        return {a: self.last_strat[i] for i, a in enumerate(self.action_list)}

    def next_strategy(self):
        # You might want to return a dictionary mapping each action in
        # `self.action_set` to the probability of picking that action
        positive_part = torch.clip(self.cum_regrets, 0, torch.inf)
        sum_regrets = torch.sum(positive_part)
        if sum_regrets <= 0:
            self.last_strat = torch.ones(self.n) / self.n
        else:
            self.last_strat = positive_part / sum_regrets
        return self.last_strategy()

    def observe_utility(self, utility):
        # assert isinstance(utility, dict) and utility.keys() == set(self.action_list)
        u = torch.tensor([utility[a] for a in self.action_list])

        # r^t = r^{t-1} + (u-<x,u>1)
        self.cum_regrets = self.cum_regrets + (u - torch.dot(self.last_strat, u))

    def reset(self, warm_start=None):
        if warm_start is not None and self.cum_regrets is not None:
            positive_part = torch.clip(self.cum_regrets, 0, torch.inf)
            sum_regrets = torch.clip(torch.sum(positive_part), 1e-3, torch.inf)
            self.cum_regrets = warm_start * positive_part / sum_regrets
            assert len(self.cum_regrets) == self.n
        else:
            self.cum_regrets = torch.zeros(self.n)
        self.last_strat = torch.ones(self.n) / self.n


class RegretMatchingPlus(RegretMatching):
    def __init__(self, action_set):
        super().__init__(action_set)

    def observe_utility(self, utility):
        # assert isinstance(utility, dict) and utility.keys() == set(self.action_list)
        u = torch.tensor([utility[a] for a in self.action_list])
        # r^t = [r^{t-1} + (u-<x,u>1)]^+
        self.cum_regrets = torch.clip(self.cum_regrets + (u - torch.dot(u, self.last_strat)), 0, torch.inf)


class DCFRRegretMatching(RegretMatching):
    def __init__(self, action_set, alpha=1.5, beta=0.0):
        super().__init__(action_set)
        self.alpha = alpha
        self.beta = beta
        self.t = 0

    def observe_utility(self, utility):
        # assert isinstance(utility, dict) and utility.keys() == set(self.action_list)
        self.t += 1

        u = torch.tensor([utility[a] for a in self.action_list])
        # <x,u>
        self.cum_regrets = self.cum_regrets + (u - torch.dot(self.last_strat, u))
        # now multiply accumulated positive regrets by t^alpha/(t^alpha + 1)
        # and negative regrets by t^beta/(t^beta + 1)
        self.cum_regrets = torch.where(
            self.cum_regrets >= 0,
            self.cum_regrets * (self.t**self.alpha / (self.t**self.alpha + 1)),
            self.cum_regrets * (self.t**self.beta / (self.t**self.beta + 1)),
        )

    def reset(self, warm_start=None):
        super().reset(warm_start=warm_start)
        self.t = 0


class PredictiveRegretMatchingPlus(RegretMatchingPlus):
    # currently prediction is the last observed utility, or 0 at step 0
    def __init__(self, action_set):
        super().__init__(action_set)
        self.prediction = torch.zeros(self.n)

    def next_strategy(self):
        # <m^{t},x^{t-1}> for m the prediction vector
        # if the first iteration, this is zero
        m_dot_x = torch.dot(self.prediction, self.last_strat)

        # theta = r^{t-1}+m^{t}-<m^{t},x^{t}>1
        theta = torch.clip(self.cum_regrets + (self.prediction - m_dot_x), 0, torch.inf)

        sum_theta = torch.sum(theta)

        if sum_theta <= 0:
            self.last_strat = torch.ones(self.n) / self.n
        else:
            self.last_strat = theta / sum_theta
        return self.last_strategy()

    def observe_utility(self, utility):
        super().observe_utility(utility)
        u = torch.tensor([utility[a] for a in self.action_list])
        self.prediction = u

    def reset(self, warm_start=None):
        super().reset(warm_start=warm_start)
        self.prediction = torch.zeros(self.n)


def solve_gamma(v, target):
    if target == 0:
        return torch.max(v)
    u = torch.sort(v, descending=True).values
    sum = 0
    sumsq = 0
    for i, ui in enumerate(u):
        sum += ui
        sumsq += ui * ui
        disc = sum * sum - (i + 1) * (sumsq - target)
        # assert disc >= 0, f"{disc}"
        disc = max(0, disc)
        gamma = (sum - torch.sqrt(torch.tensor(disc))) / (i + 1)
        if i == len(u) - 1 or gamma >= u[i + 1]:
            return gamma
    assert False


class IRPRMPlus(RegretMatchingPlus):
    def __init__(self, action_set):
        super().__init__(action_set)
        self.prediction = torch.zeros(self.n)
        self.last_u = torch.zeros(self.n)

    def next_strategy(self):
        if self.cum_regrets.max() == 0:
            return self.last_strategy()

        old_norm = torch.linalg.norm(torch.clip(self.cum_regrets, 0, torch.inf))
        self.cum_regrets += self.prediction
        gamma = solve_gamma(self.cum_regrets, old_norm**2)
        self.cum_regrets -= gamma
        regrets = torch.clip(self.cum_regrets, 0, torch.inf)

        regret_sum = torch.sum(regrets)
        if regret_sum <= 0:
            self.last_strat = torch.ones(self.n) / self.n
        else:
            self.last_strat = regrets / regret_sum
        return self.last_strategy()

    def observe_utility(self, utility):
        u = torch.tensor([utility[a] for a in self.action_list])
        u_p = u - self.prediction
        self.cum_regrets = self.cum_regrets + u_p - u_p @ self.last_strat
        self.cum_regrets = torch.clip(self.cum_regrets, 0, torch.inf)

        self.prediction = u

    def reset(self, warm_start=None):
        super().reset(warm_start=warm_start)
        self.prediction = torch.zeros(self.n)
