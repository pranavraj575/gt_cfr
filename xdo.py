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

    def final_expansion(self):
        for player, single_player_tree in self.single_player_trees.items():
            for infoset_id, dic in single_player_tree.items():
                self.maybe_add_regret_minimizer(player=player,
                                                infoset_id=infoset_id,
                                                actions=dic[LEGAL_ACTIONS],
                                                force_reset=True,
                                                )

    def assert_regret_minimizers_have_correct_support(self, min_restricted_actions=1,
                                                      max_restricted_actions=float('inf')):
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

    def iterate_nodes(self, node=None):
        """
        debug method, iterate over nodes
        """
        if node is None:
            node = self.root
        yield node
        actions = []
        if (not node.terminal) and (not node.is_chance_node()):
            rm = self.player_to_regret_minimizers[node.player][node.infoset_id]
            for a in rm.action_set:
                if a in node.children:
                    actions.append(a)
        elif node.is_chance_node():
            actions = list(node.children.keys())

        for a in actions:
            for n in self.iterate_nodes(node.children[a]):
                yield n


def plt_all(game_name, game_tag=''):
    for log_scale in False, True:
        for clock_time in True, False:
            plt_one(game_name, game_tag=game_tag, log_scale=log_scale, clock_time=clock_time)


def plt_one(game_name, game_tag='', log_scale=False, clock_time=False):
    key = game_name + game_tag
    DIR = os.path.dirname(__file__)
    save_path = os.path.join(DIR, 'output', )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, key, 'xdo_metrics.pkl')
    if os.path.exists(save_file):
        f = open(save_file, 'rb')
        gap_over_time = pickle.load(f)
        f.close()
        if log_scale:
            fig, ax = plt.subplots()
            ax.set_yscale("log", nonpositive='mask')

        for t in gap_over_time:
            if clock_time:
                plt.plot(gap_over_time[t]['times'], gap_over_time[t]['conv'], label='xdo with ' + t)
            else:
                plt.plot(gap_over_time[t]['conv'], label='xdo with ' + t)

        save_file = os.path.join(save_path, key, 'gtcfr_conv_by_time.npy')
        if os.path.exists(save_file):
            gap_over_time = np.load(save_file)
            if clock_time:
                plt.plot(gap_over_time['times'], gap_over_time['conv'], label='gtcfr')
            else:
                plt.plot(gap_over_time['conv'], label='gtcfr')
        plt.ylabel('nash conv')
        plt.title(key)
        if clock_time:
            plt.xlabel('clock time')
        else:
            plt.xlabel('epochs')
        plt.legend()
        fn = os.path.join(save_path, key, 'xdo_gtcfr_cmp_plt' + ('_log' if log_scale else '') + (
            '_clock' if clock_time else '') + '.png')
        plt.savefig(fn)
        print('saving to', fn)
        # plt.show()
        plt.close()

        max_epoch = -float('inf')
        for t in gap_over_time:
            iss = gap_over_time[t]['expanded_infosets']
            ass = gap_over_time[t]['all_infosets']
            max_epoch = max(max_epoch, len(iss) - 1)
            plt.plot(np.sum(iss, axis=1)/np.sum(ass), label='xdo with ' + t)
        plt.legend()
        plt.plot([0, max_epoch], [1, 1], linestyle='--', alpha=.5)
        plt.ylabel('proportion of infosets expanded')
        plt.xlabel('epochs')
        plt.title(key)
        fn = os.path.join(save_path, key, 'xdo_expansion_plt.png')
        plt.savefig(fn)
        print('saving to', fn)
        # plt.show()
        plt.close()


def universal_poker_tag_to_args(tag):
    dic = {'numRanks': 6, 'numSuits': 4, 'bettingAbstraction': 'fcpa', "stack": '1200 1200', 'blind': '100 100'}
    if '-small_deck' in tag:
        dic['numRanks'] = 3
        dic['numSuits'] = 2

    if '-large_deck' in tag:
        dic['numRanks'] = 13

    if '-include_half' in tag:
        dic['bettingAbstraction'] = 'fchpa'

    if '-2_hole' in tag:
        dic['numHoleCards'] = 2

    if '-3_rounds' in tag:
        dic['numBoardCards'] = '1 1'

    if '-4_rounds' in tag:
        dic['numBoardCards'] = '1 1 1'
    return dic


def main(game_name, tag, game_tag='', overwrite=False):
    key = game_name + game_tag
    DIR = os.path.dirname(__file__)
    save_path = os.path.join(DIR, 'output', key, )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, 'xdo_metrics.pkl')

    if os.path.exists(save_file):
        f = open(save_file, 'rb')
        gap_over_time = pickle.load(f)
        f.close()
    else:
        gap_over_time = dict()

    if (not overwrite) and tag in gap_over_time:
        return

    if tag == 'CFR':
        RM = RegretMatching
    elif tag == 'CFR+':
        RM = RegretMatchingPlus
    elif tag == 'DCFR':
        RM = DCFRRegretMatching
    elif tag == 'PCFR':
        RM = PredictiveRegretMatchingPlus
    else:
        raise Exception(tag)
    args = dict()
    if game_name == 'universal_poker':
        args = universal_poker_tag_to_args(game_tag)
    game = pyspiel.load_game(game_name,
                             args
                             )

    xdo = XDO(
        root_state=PyspielStateStructure(game.new_initial_state()),
        rm_class=RM,
    )
    print('expanded nodes', xdo.count_nodes())
    # should only be one available action per infoset
    xdo.assert_regret_minimizers_have_correct_support(max_restricted_actions=1)
    print(xdo.constant_sum_nash_gap(player_strategies={0: xdo.obtain_strategy(0), 1: xdo.obtain_strategy(1)},
                                    sequential_form=False))
    update_times = []
    nash_gaps = []
    expanded_infosets = []
    i = 0
    for ii in range(20):
        sum_sq_0 = dict()
        sum_sq_1 = dict()
        accumulated_weight = 0.
        value0 = None
        value1 = None
        avg_sq_0 = None
        avg_sq_1 = None
        start = time.time()

        for _ in range(1, 401):
            i += 1
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
        br0, bru_0 = xdo.best_response_strategy(player=0, other_player_strategies={1: xdo.obtain_strategy(1)},
                                                sequential_form=False)
        p0_updated = xdo.add_support_of_strategy(player=0, strategy=br0)
        br1, bru_1 = xdo.best_response_strategy(player=1, other_player_strategies={0: xdo.obtain_strategy(0)},
                                                sequential_form=False)
        p1_updated = xdo.add_support_of_strategy(player=1, strategy=br1)
        any_updates = p0_updated or p1_updated
        if any_updates:
            xdo.reset_regret_minimizers(warm_start=1.)
            i = 0
            # restart regret calculation from scratch, as there is a new restricted game
        gap = xdo.constant_sum_nash_gap(player_strategies={0: avg_sq_0, 1: avg_sq_1}, sequential_form=True)
        nash_gaps.append(gap)

        expanded_infos = xdo.count_infosets()
        expanded_infosets.append(expanded_infos)
        print('expanded infosets', expanded_infos)
        print(ii, gap)
        print('updates', any_updates)
        print('p0 val, br val, improvement', value0, bru_0, bru_0 - value0)
        print('p1 val, br val, improvement', value1, bru_1, bru_1 - value1)
        print()
    xdo.final_expansion()
    all_infosets = xdo.count_infosets()

    gap_over_time[tag] = {'times': np.cumsum(update_times),
                          'conv': nash_gaps,
                          'expanded_infosets': np.array(expanded_infosets),
                          'all_infosets': np.array(all_infosets)
                          }
    f = open(save_file, 'wb')
    pickle.dump(gap_over_time, f)
    f.close()


if __name__ == '__main__':
    from gt_cfr import PyspielStateStructure
    import time, os, pickle
    import matplotlib.pyplot as plt

    overwrite = False
    game_name = 'universal_poker'
    game_tag = '-small_deck-2_hole'

    if game_name != 'universal_poker':
        game_tag = ''
    from regret_minimizer import RegretMatching, RegretMatchingPlus, DCFRRegretMatching, PredictiveRegretMatchingPlus

    tags = ['CFR', 'CFR+', 'DCFR', 'PCFR']

    for tag in tags:
        plt_all(game_name=game_name, game_tag=game_tag, )
        np.random.seed(21)
        main(game_name, tag=tag, game_tag=game_tag, overwrite=overwrite)

    plt_all(game_name=game_name, game_tag=game_tag, )

    # plt.plot(gap_over_time[tag][0], gap_over_time[tag][1])
    # plt.show()
