import unittest
from copy import copy

import numpy as np

from landlordai.game.card import Card
from landlordai.game.landlord import LandlordGame
from landlordai.game.player import LearningPlayer_v1, TurnPosition


class TestLandlordMethods(unittest.TestCase):
    def test_player_move(self):
        players = [LearningPlayer_v1(name='random')] * 3
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4 + [Card.KING] * 4 + [Card.QUEEN] * 4 + [Card.JACK] * 4 + [Card.THREE],
            TurnPosition.SECOND: [Card.TEN] * 4 + [Card.NINE] * 4 + [Card.EIGHT] * 4 + [Card.SEVEN] * 4 + [Card.THREE],
            TurnPosition.THIRD: [Card.FIVE] * 4 + [Card.FOUR] * 4 + [Card.SIX] * 4 + [Card.TWO] * 4 + [Card.THREE] * 2 + [Card.LITTLE_JOKER] + [Card.BIG_JOKER]
        }
        game.betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 3)
        game2 = copy(game)
        best_move = players[0].make_move(game, game.get_current_position())
        game.play_move(best_move)
        self.assertFalse(game2.get_hand(TurnPosition.THIRD) == game.get_hand(TurnPosition.THIRD))

    def test_player_game(self):
        players = [LearningPlayer_v1(name='random')] * 3
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4 + [Card.KING] * 4 + [Card.QUEEN] * 4 + [Card.JACK] * 4 + [Card.THREE],
            TurnPosition.SECOND: [Card.TEN] * 4 + [Card.NINE] * 4 + [Card.EIGHT] * 4 + [Card.SEVEN] * 4 + [Card.THREE],
            TurnPosition.THIRD: [Card.FIVE] * 4 + [Card.FOUR] * 4 + [Card.SIX] * 4 + [Card.TWO] * 4 + [Card.THREE] * 2 + [Card.LITTLE_JOKER] + [Card.BIG_JOKER]
        }
        game.betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 3)
        game.main_game()
        self.assertTrue(np.sum(np.abs(game.get_scores())) > 0)
        # game is over
        self.assertTrue(np.abs(players[0]._record_future_q[-1]) > 0.5)

        self.assertTrue(np.allclose(players[0].derive_features(game)[:len(game.get_move_logs()) - 1],
                                    players[0].derive_record_features(game)[0][:len(game.get_move_logs()) - 1]))
        self.assertTrue(np.allclose(players[0].derive_features(game)[len(game.get_move_logs()) - 1],
                                    players[0].derive_record_features(game)[1]))

        features = players[0].derive_features(game)
        self.assertTrue(np.sum(features[:, players[0].get_feature_index('I_AM_LANDLORD')]) != 0)
        # it is possible this guy never plays, eventually
        self.assertTrue(np.sum(features[:, players[0].get_feature_index('I_AM_BEFORE_LANDLORD')]) != 0)

    def test_full_game(self):
        players = [LearningPlayer_v1(name='random') for _ in range(3)]
        game = LandlordGame(players=players)
        game.play_round()

        while np.sum(np.abs(game.get_scores())) == 0:
            players = [LearningPlayer_v1(name='random') for _ in range(3)]
            game = LandlordGame(players=players)
            game.play_round()

        # game is over
        for i in range(3):
            #print(players[i].record_future_q[-1])
            #self.assertTrue(np.abs(players[i].record_future_q[-1]) > 0.5)

            self.assertTrue(np.allclose(players[i].derive_features(game)[:len(game.get_move_logs()) - 1],
                                        players[i].derive_record_features(game)[0][:len(game.get_move_logs()) - 1]))
            self.assertTrue(np.allclose(players[i].derive_features(game)[len(game.get_move_logs()) - 1],
                                        players[i].derive_record_features(game)[1]))

            features = players[i].derive_features(game)
            self.assertTrue(np.sum(features[:, players[i].get_feature_index('I_AM_LANDLORD')]) != 0)
            # it is possible this guy never plays, eventually
            self.assertTrue(np.sum(features[:, players[i].get_feature_index('I_AM_BEFORE_LANDLORD')]) != 0)

    def test_llord_winning(self):
        players = [LearningPlayer_v1(name='random')] * 3
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4 + [Card.KING] * 4 + [Card.QUEEN] * 4 + [Card.JACK] * 4 + [Card.THREE],
            TurnPosition.SECOND: [Card.TEN] * 4 + [Card.NINE] * 4 + [Card.EIGHT] * 4 + [Card.SEVEN] * 4 + [Card.THREE],
            TurnPosition.THIRD: [Card.FIVE] * 4
        }
        game.betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 3)
        game.main_game()
        self.assertTrue(TurnPosition.THIRD in game.get_winners())
        self.assertTrue(len(game.get_move_logs()) == 1)

    def test_peasant_winning(self):
        for i in range(10):
            players = [LearningPlayer_v1(name='random')] * 3
            game = LandlordGame(players=players)
            hands = {
                TurnPosition.FIRST: [Card.ACE] * 4,
                TurnPosition.SECOND: [Card.TEN] + [Card.THREE],
                TurnPosition.THIRD: [Card.FIVE] * 3 + [Card.THREE] + [Card.FOUR]
            }
            game.betting_complete = True
            game.force_setup(TurnPosition.THIRD, hands, 3)
            hand_vector = players[0].get_hand_vector(game, TurnPosition.FIRST)
            self.assertTrue(hand_vector[11] == 4)
            self.assertTrue(hand_vector[-2] == 2)
            self.assertTrue(hand_vector[-3] == 5)
            self.assertTrue(hand_vector[-1] == 4)
            #self.assertTrue(np.sum(hand_vector) == 4)
            game.main_game()
            self.assertTrue(TurnPosition.THIRD not in game.get_winners())
            self.assertTrue(TurnPosition.SECOND in game.get_winners())
            self.assertTrue(TurnPosition.FIRST in game.get_winners())
            self.assertTrue(len(game.get_move_logs()) == 2)

if __name__ == '__main__':
    unittest.main()

