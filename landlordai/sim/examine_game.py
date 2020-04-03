from landlordai.game.landlord import LandlordGame
from landlordai.game.player import LearningPlayer_v1, RandomPlayer
from landlordai.sim.game_stats import GameStats
from landlordai.sim.simulate import Simulator

import numpy as np
from tqdm import tqdm

def load_net(net):
    return LearningPlayer_v1(name=net, net_dir='../models/' + net,
                             use_montecarlo_random=False,
                             mc_best_move_depth=4,
                             epsilon=0,
                             learning_rate=0.2)

if __name__ == "__main__":
    #players = [load_net('3_30_sim5_model8')] + [load_net('3_30_sim5_model14')] + [LearningPlayer_v1(name='random') for _ in range(1)]
    #players = [LearningPlayer_v1(name='random') for _ in range(3)]
    #players = [load_net('4_1_sim1_model5'), load_net('4_1_sim1_model0')] + [LearningPlayer_v1(name='random') for _ in range(1)]
    players = [load_net('4_2_sim4_model7'), load_net('4_2_sim3_model0'), load_net('4_2_sim3_model5')]


    while True:
        game = LandlordGame(players=players)
        game.play_round(debug=True)
        if game.has_winners():
            break

    print('\n')
    for i in range(3):
        print(players[i].get_future_q())
        print(players[i]._record_state_q)
        print('\n')

    print('')



