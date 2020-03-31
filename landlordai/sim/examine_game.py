from landlordai.game.landlord import LandlordGame
from landlordai.game.player import LearningPlayer_v1, RandomPlayer
from landlordai.sim.game_stats import GameStats
from landlordai.sim.simulate import Simulator

import numpy as np

def load_net(net):
    return LearningPlayer_v1(name=net, net_dir='../models/' + net)

if __name__ == "__main__":
    #players = [load_net('3_30_sim5_model8')] + [load_net('3_30_sim5_model14')] + [LearningPlayer_v1(name='random') for _ in range(1)]
    players = [load_net('3_30_sim5_model15') for _ in range(3)]

    while True:
        game = LandlordGame(players=players)
        game.play_round()
        if game.has_winners():
            break
        else:
            print('Ended at Draw')

    print('\n')
    for i in range(3):
        print(players[i].get_future_q())
        print(players[i].get_record_hand_vectors())
        print('\n')

    print('')



