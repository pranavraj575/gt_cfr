class RegretMinimizer(object):
    def next_strategy(self):
        raise NotImplementedError
    def observe_utility(self, utility):
        raise NotImplementedError
class RegretMatching(RegretMinimizer):
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

class RegretMatchingPlus(RegretMinimizer):
    def __init__(self, action_set):
        self.action_set = set(action_set)
        self.cum_regrets = {a: 0. for a in self.action_set}
        self.last_strat = None

    def next_strategy(self):
        sum_regrets = sum(self.cum_regrets[a] for a in self.action_set)  # cumulative regrets will be nonnegative since this is RM+
        if sum_regrets == 0:
            self.last_strat = {
                a: 1/len(self.action_set)
                for a in self.action_set
            }
        else:
            self.last_strat = {
                a: self.cum_regrets[a]/sum_regrets  # cumulative regrets will be nonnegative since this is RM+
                for a in self.action_set
            }
        return self.last_strat.copy()

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set

        # <x,u>
        ute = sum(self.last_strat[a]*utility[a] for a in self.action_set)

        # r^t = [r^{t-1} + (u-<x,u>1)]^+
        self.cum_regrets = {
            # instant regret is u-<x,u>1, at dimension a this is u[a]-<x,u>
            a: max(self.cum_regrets[a] + (utility[a] - ute), 0)
            for a in self.action_set
        }


class DCFRRegretMatching(RegretMatching):
    def __init__(self, action_set, alpha=1.5, beta=0.):
        super().__init__(action_set)
        self.alpha = alpha
        self.beta = beta
        self.t = 0

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set
        self.t += 1

        # <x,u>
        ute = sum(self.last_strat[a]*utility[a] for a in self.action_set)

        self.cum_regrets = {
            # instant regret is u-<x,u>1, at dimension a this is u[a]-<x,u>
            a: max(self.cum_regrets[a] + (utility[a] - ute), 0)
            for a in self.action_set
        }
        # now multiply accumulated positive regrets by t^alpha/(t^alpha + 1)
        # and negative regrets by t^beta/(t^beta + 1)
        self.cum_regrets = {
            a: (r*self.t**self.alpha/(self.t**self.alpha + 1) if r >= 0 else
                r*self.t**self.beta/(self.t**self.beta + 1))
            for a, r in self.cum_regrets.items()
        }


class PredictiveRegretMatchingPlus(RegretMatchingPlus):
    # currently prediction is the last observed utility, or 0 at step 0
    def __init__(self, action_set):
        super().__init__(action_set)
        self.prediction = {a: 0. for a in self.action_set}

    def next_strategy(self):
        # <m^{t},x^{t-1}> for m the prediction vector
        # if the first iteration, set this to zero
        if self.last_strat is None:
            m_dot_x = 0.
        else:
            m_dot_x = sum(
                self.prediction[a]*self.last_strat[a]
                for a in self.action_set
            )

        # theta = r^{t-1}+m^{t}-<m^{t},x^{t}>1
        theta = {
            a: max(self.cum_regrets[a] + self.prediction[a] - m_dot_x, 0)
            for a in self.action_set
        }
        sum_theta = sum(v for a, v in theta.items())

        if sum_theta == 0:
            self.last_strat = {
                a: 1/len(self.action_set)
                for a in self.action_set
            }
        else:
            self.last_strat = {
                a: v/sum_theta
                for a, v in theta.items()
            }
        return self.last_strat.copy()

    def observe_utility(self, utility):
        super().observe_utility(utility)
        self.prediction = utility

