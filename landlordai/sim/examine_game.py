from landlordai.game.landlord import LandlordGame
from landlordai.game.player import LearningPlayer_v1, RandomPlayer
from landlordai.sim.game_stats import GameStats
from landlordai.sim.simulate import Simulator

if __name__ == "__main__":
    players = [
        LearningPlayer_v1(name='3_29_sim10_model' + str(i), net_dir='../models/3_29_sim10_model' + str(i))
    for i in range(1)] + [
        LearningPlayer_v1(name='3_29_sim11_model' + str(i), net_dir='../models/3_29_sim11_model' + str(i))
    for i in range(1)] + [LearningPlayer_v1(name='random') for _ in range(1)]


    game = LandlordGame(players=players)
    while True:
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



