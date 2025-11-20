import numpy as np
import pyspiel

from regret_minimizer import RegretMinimizer, RegretMatchingPlus
from gt_cfr import (GTCFR, update_and_produce_avg_strats, PARENT_SEQUENCE, LEGAL_ACTIONS)
from state_structure import StateStructure

RESTRICTED = 'restricted_actions'


class XDO(GTCFR):
    def __init__(self, root_state: StateStructure,
                 rm_class: type(RegretMinimizer) = RegretMatchingPlus,
                 rm_kwargs=None):
        super().__init__(root_state=root_state, rm_class=rm_class, rm_kwargs=rm_kwargs)
        self.create_full_tree()
        self.initial_expansion()

    def initial_expansion(self):
        # force each regret minimizer to have exactly one choice in action, chosen uniformly at random
        for player, single_player_tree in self.single_player_trees.items():
            for infoset_id, dic in single_player_tree.items():
                self.maybe_add_regret_minimizer(player=player,
                                                infoset_id=infoset_id,
                                                actions=[np.random.choice(dic[LEGAL_ACTIONS])],
                                                force_reset=True,
                                                )

    def assert_regret_minimizers_have_correct_support(self, min_restricted_actions=1, max_restricted_actions=float('inf')):
        for player, single_player_tree in self.single_player_trees.items():
            for infoset_id, dic in single_player_tree.items():
                restricted = self.player_to_regret_minimizers[player][infoset_id].action_set
                assert len(restricted) >= min_restricted_actions
                assert len(restricted) <= max_restricted_actions

    def add_support_of_strategy(self, player, strategy, epsilon=0, sequence_form=False):
        """
        adds support of a strategy (where the probability of (action | infoset) > epsilon)
        :param player:
        :param strategy:
        :param epsilon:
        :return:
        """
        any_updates = False
        single_player_tree = self.single_player_trees[player]
        support = dict()
        if sequence_form:
            for seq in strategy:
                infoset_id, a = seq
                parent_seq = single_player_tree[infoset_id][PARENT_SEQUENCE]
                prob_flow = 1.
                if parent_seq is not None:
                    prob_flow = strategy[parent_seq]
                if strategy[seq] > epsilon*prob_flow:
                    # strategy(seq)/prob_flow > epsilon, but dodge the /0 error
                    if infoset_id not in support:
                        support[infoset_id] = set()
                    support[infoset_id].add(a)
        else:
            for infoset_id in strategy:
                support[infoset_id] = {a for a in strategy[infoset_id] if strategy[infoset_id][a] > epsilon}
        for infoset_id in support:
            regret_min = self.player_to_regret_minimizers[player][infoset_id]
            if len(support[infoset_id].difference(regret_min.action_set)) > 0:
                # if anything in support of strategy is not in action set already
                full_support = support[infoset_id].union(regret_min.action_set)
                self.maybe_add_regret_minimizer(player=player,
                                                infoset_id=infoset_id,
                                                actions=full_support,
                                                force_reset=True,
                                                )
                any_updates = True
        return any_updates


if __name__ == '__main__':
    from gt_cfr import PyspielStateStructure
    import time, os
    import matplotlib.pyplot as plt

    np.random.seed(21)
    game_name = 'leduc_poker'
    game = pyspiel.load_game(game_name)

    xdo = XDO(
        root_state=PyspielStateStructure(game.new_initial_state()),
        rm_class=RegretMatchingPlus,
    )
    print('expanded nodes', xdo.count_nodes())
    # should only be one available action per infoset
    xdo.assert_regret_minimizers_have_correct_support(max_restricted_actions=1)
    print(xdo.constant_sum_nash_gap(player_strategies={0: xdo.obtain_strategy(0), 1: xdo.obtain_strategy(1)}, sequential_form=False))
    update_times = []
    nash_gaps = []
    for _ in range(100):

        sum_sq_0 = dict()
        sum_sq_1 = dict()
        accumulated_weight = 0.
        value0 = None
        value1 = None
        avg_sq_0 = None
        avg_sq_1 = None
        start = time.time()
        for i in range(100):
            bhv_0 = xdo.obtain_strategy(player=0)
            bhv_1 = xdo.obtain_strategy(player=1)
            x0 = xdo.convert_to_sequence_form(player=0, behavioral_strat=bhv_0)
            x1 = xdo.convert_to_sequence_form(player=1, behavioral_strat=bhv_1)

            u0, obs_infosets0 = xdo.compute_utilities(player=0, other_player_strategies={1: bhv_1})
            u1, obs_infosets1 = xdo.compute_utilities(player=1, other_player_strategies={0: bhv_0})
            xdo.observe_utility(player=0, utility=u0, observed_infosets=obs_infosets0)
            xdo.observe_utility(player=1, utility=u1, observed_infosets=obs_infosets1)
            avg_sq_0, avg_sq_1, w = update_and_produce_avg_strats(
                gtcfr=xdo,
                accumulated_weight=accumulated_weight,
                sum_sq_0=sum_sq_0, sum_sq_1=sum_sq_1,
                x0=x0, x1=x1,
                b0=bhv_0, b1=bhv_1,
                i=i, )
            accumulated_weight += w
        update_times.append(time.time() - start)
        value0 = xdo.compute_player_value(player=0, player_sequential_strategies={0: avg_sq_0, 1: avg_sq_1})
        value1 = xdo.compute_player_value(player=1, player_sequential_strategies={0: avg_sq_0, 1: avg_sq_1})
        br0, bru_0 = xdo.best_response_strategy(player=0, other_player_strategies={1: xdo.obtain_strategy(1)}, sequential_form=False)
        p0_updated = xdo.add_support_of_strategy(player=0, strategy=br0)
        br1, bru_1 = xdo.best_response_strategy(player=1, other_player_strategies={0: xdo.obtain_strategy(0)}, sequential_form=False)
        p1_updated = xdo.add_support_of_strategy(player=1, strategy=br1)
        any_updates = p0_updated or p1_updated
        if any_updates:
            xdo.reset_regret_minimizers()
            # restart regret calculation from scratch, as there is a new restricted game
        gap = xdo.constant_sum_nash_gap(player_strategies={0: avg_sq_0, 1: avg_sq_1}, sequential_form=True)
        nash_gaps.append(gap)
        print(gap)
        print('updates', any_updates)
        print('p0 val and br val', value0, bru_0)
        print('p1 val and br val', value1, bru_1)
        print()

    DIR = os.path.dirname(__file__)
    save_path = os.path.join(DIR, 'output', )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, 'nxdo_' + game_name + '_conv_by_time')
    gap_over_time = np.array([np.cumsum(update_times),
                              nash_gaps])
    np.save(save_file, gap_over_time)
    plt.plot(gap_over_time[0], gap_over_time[1])
    plt.show()
