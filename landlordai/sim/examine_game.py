from landlordai.game.landlord import LandlordGame
from landlordai.game.player import LearningPlayer_v1, RandomPlayer
from landlordai.sim.game_stats import GameStats
from landlordai.sim.simulate import Simulator

import numpy as np
from tqdm import tqdm

def load_net(net):
    return LearningPlayer_v1(name=net, net_dir='../models/' + net)

if __name__ == "__main__":
    #players = [load_net('3_30_sim5_model8')] + [load_net('3_30_sim5_model14')] + [LearningPlayer_v1(name='random') for _ in range(1)]
    #players = [LearningPlayer_v1(name='random') for _ in range(3)]
    players = [load_net('3_31_sim4_model6'), load_net('3_31_sim4_model0'), load_net('3_31_sim4_model0')]


    while True:
        game = LandlordGame(players=players)
        game.play_round(debug=True)
        if game.has_winners():
            break

    print('\n')
    for i in range(3):
        print(players[i].get_future_q())
        print(players[i].get_record_hand_vectors())
        print('\n')

    print('')



