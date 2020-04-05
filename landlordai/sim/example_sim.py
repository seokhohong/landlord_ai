from landlordai.game.player import LearningPlayer
from landlordai.sim.game_stats import GameStats
from landlordai.sim.simulate import Simulator

from tqdm import tqdm
from tqdm import tqdm

from landlordai.game.player import LearningPlayer
from landlordai.sim.game_stats import GameStats
from landlordai.sim.simulate import Simulator


def load_net(net):
    return LearningPlayer(name=net, net_dir='../models/' + net, estimation_mode=LearningPlayer.MONTECARLO_RANDOM)

if __name__ == "__main__":
    '''
    players = [
        LearningPlayer_v1(name='3_29_sim8_model' + str(i), net_dir='../models/3_29_sim8_model' + str(i))
    for i in range(3)] + [LearningPlayer_v1(name='random') for _ in range(3)] + \
              [LearningPlayer_v1(name='3_29_sim7_model' + str(i), net_dir='../models/3_29_sim7_model' + str(i))
    for i in range(3)]
    '''
    players = [LearningPlayer('random', estimation_mode=LearningPlayer.MONTECARLO_RANDOM) for i in range(1)] + \
              [load_net('4_1_sim3_model2'), load_net('4_1_sim4_model4'), load_net('4_1_sim4_model6'),
             load_net('4_1_sim3_model3'), load_net('4_1_sim3_model0')]

    for i in tqdm(range(1)):
        simulator = Simulator(20, players)
        simulator.play_rounds()

        results = simulator.get_results()

    stats = GameStats(players, results)

    unique_names = set([player.get_name() for player in players])

    for name in unique_names:
        print(name, stats.get_win_rate(name), stats.get_elo(name))

    #game = LandlordGame(players=players)
    #game.play_round()
    #for turn, move in game.get_move_logs():
    #    print(turn, move)

