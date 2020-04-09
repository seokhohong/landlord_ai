from landlordai.game.landlord import LandlordGame
from landlordai.game.player import LearningPlayer, RandomPlayer, LearningPlayer_v2
from landlordai.sim.game_stats import GameStats
from landlordai.sim.simulate import Simulator

import numpy as np
from tqdm import tqdm

def load_net(net):
    return LearningPlayer(name=net, net_dir='../models/' + net,
                          estimation_mode=LearningPlayer.ACTUAL_Q,
                          estimation_depth=7,
                          discount_factor=1,
                          epsilon=0,
                          learning_rate=0.3)

def load_v2_net(net):
    return LearningPlayer_v2(name=net, net_dir='../models/' + net,
                          estimation_mode=LearningPlayer.ACTUAL_Q,
                          estimation_depth=7,
                          discount_factor=1,
                          epsilon=0,
                          learning_rate=0.3)

if __name__ == "__main__":
    players = [load_net('4_9_actualq1_model30'),
               load_net('4_2_sim4_model14'),
               load_net('4_7_actualq5_model20')]

    while True:
        game = LandlordGame(players=players)
        game.play_round(debug=True)
        if game.has_winners():
            break

    print('\n')
    for i in range(3):
        players[i].compute_future_q(game)
        print(players[i].get_name())
        print(np.array(players[i].get_estimated_qs(), dtype=np.float16))
        print(np.array(players[i]._record_state_q, dtype=np.float16))
        print('\n')

    print('')



