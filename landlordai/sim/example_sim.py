from landlordai.game.landlord import LandlordGame
from landlordai.game.player import LearningPlayer_v1, RandomPlayer
from landlordai.sim.game_stats import GameStats
from landlordai.sim.simulate import Simulator

import numpy as np

if __name__ == "__main__":
    '''
    players = [
        LearningPlayer_v1(name='3_29_sim8_model' + str(i), net_dir='../models/3_29_sim8_model' + str(i))
    for i in range(3)] + [LearningPlayer_v1(name='random') for _ in range(3)] + \
              [LearningPlayer_v1(name='3_29_sim7_model' + str(i), net_dir='../models/3_29_sim7_model' + str(i))
    for i in range(3)]
    '''
    players = [LearningPlayer_v1('random') for i in range(3)]

    simulator = Simulator(1, players)
    simulator.play_rounds()

    results = simulator.get_result_pairs()

    stats = GameStats(players, results)

    unique_names = set([player.get_name() for player in players])

    for name in unique_names:
        print(name, stats.get_win_rate(name))

    #game = LandlordGame(players=players)
    #game.play_round()
    #for turn, move in game.get_move_logs():
    #    print(turn, move)

