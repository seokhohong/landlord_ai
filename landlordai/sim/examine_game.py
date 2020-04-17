from landlordai.game.landlord import LandlordGame
from landlordai.game.player import LearningPlayer, RandomPlayer, LearningPlayer_v2
from landlordai.sim.game_stats import GameStats
from landlordai.sim.simulate import Simulator

import numpy as np
from tqdm import tqdm

def load_net(net, models_dir='../models/'):
    return LearningPlayer(name=net, net_dir=models_dir + net,
                          estimation_mode=LearningPlayer.ACTUAL_Q,
                          discount_factor=1,
                          epsilon=0,
                          learning_rate=0.3)

def load_v2_net(net, models_dir='../models/'):
    return LearningPlayer_v2(name=net, net_dir=models_dir + net,
                          estimation_mode=LearningPlayer.ACTUAL_Q,
                          estimation_depth=7,
                          discount_factor=1,
                          epsilon=0,
                          learning_rate=0.3)

if __name__ == "__main__":
    #players = [load_net('4_11_actualq4_model20'),
    #           load_net('4_11_actualq4_model20'),
    #           load_net('4_11_actualq4_model20')]

    players = [
        load_v2_net('4_13_stream2_model3_170', '../stream_models/'),
        load_v2_net('4_13_stream2_model2_194', '../stream_models/'),
        load_v2_net('4_13_stream2_model1_141', '../stream_models/')
    ]

    while True:
        game = LandlordGame(players=players)
        game.play_round(debug=True)
        if game.has_winners():
            break

    def printout_floats(array):
        print(', '.join(["%.3f" % val for val in array]))

    print('\n')
    for i in range(3):
        players[i].compute_future_q(game)
        print(players[i].get_name())
        printout_floats(players[i].get_estimated_qs())
        printout_floats(players[i]._record_state_q)
        print('\n')

    print('')



