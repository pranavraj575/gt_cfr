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
        # FINISH

    def next_strategy(self):
        # FINISH
        # You might want to return a dictionary mapping each action in
        # `self.action_set` to the probability of picking that action
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
        # FINISH