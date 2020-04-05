import cProfile
from pstats import SortKey
import pstats

from landlordai.game.player import LearningPlayer
from landlordai.sim.game_stats import GameStats
from landlordai.sim.simulate import Simulator

if __name__ == '__main__':

    def load_net(net):
        return LearningPlayer(name=net, net_dir='../models/' + net, estimation_mode=LearningPlayer.CONSENSUS_Q)

    players = [LearningPlayer('random', estimation_mode=LearningPlayer.MONTECARLO_RANDOM) for i in range(1)] + \
              [load_net('4_1_sim3_model2'), load_net('4_1_sim4_model4'), load_net('4_1_sim4_model6'),
             load_net('4_1_sim3_model3'), load_net('4_1_sim3_model0')]

    pr = cProfile.Profile()
    pr.enable()

    simulator = Simulator(1, players)
    simulator.play_rounds()

    pr.disable()
    pr.create_stats()

    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats()