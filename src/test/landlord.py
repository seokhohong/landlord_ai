from src.game.card import Card
from src.game.landlord import LandlordGame
from src.game.move import SpecificMove, RankedMoveType, MoveType, BetMove
from src.game.player import RandomPlayer, NoBetPlayer, LearningPlayer_v1, TurnPosition
from collections import Counter
import numpy as np
import unittest
from copy import copy

class TestLandlordMethods(unittest.TestCase):
    def test_basic_game(self):
        game = LandlordGame(players=[RandomPlayer(name='random')] * 3)
        game.play_round()
        for player in list(TurnPosition):
            if player in game.winners:
                self.assertTrue(game.get_r_from_perspective(player) > 0)
            else:
                self.assertTrue(game.get_r_from_perspective(player) < 0)

    def test_extended_game(self):
        game = LandlordGame(players=[RandomPlayer(name='random')] * 3)
        game.play_round()
        game2 = copy(game)
        self.assertFalse(game.get_move_logs() == game2.get_move_logs())
        self.assertTrue(game.get_hand(TurnPosition.SECOND) == game2.get_hand(TurnPosition.SECOND))
        self.assertTrue(game.get_last_played() == game2.get_last_played())

    def test_many_games(self):
        for i in range(10):
            game = LandlordGame(players=[RandomPlayer(name='random')] * 3)
            game.play_round()

    def test_game_copy(self):
        game = LandlordGame(players=[RandomPlayer(name='random')] * 3)
        game2 = copy(game)
        game.play_round()

        self.assertFalse(game.get_move_logs() == game2.get_move_logs())
        self.assertFalse(game.get_hand(TurnPosition.SECOND) == game2.get_hand(TurnPosition.SECOND))
        self.assertFalse(game.get_last_played() == game2.get_last_played())

    def test_nobet_game(self):
        game = LandlordGame(players=[NoBetPlayer(name='random')] * 3)
        game.play_round()
        self.assertTrue(np.sum(np.abs(game.scores)) == 0)

    def test_setup(self):
        players = [LearningPlayer_v1('v1', None)] * 3
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4 + [Card.KING] * 4 + [Card.QUEEN] * 4 + [Card.JACK] * 4 + [Card.THREE],
            TurnPosition.SECOND: [Card.TEN] * 4 + [Card.NINE] * 4 + [Card.EIGHT] * 4 + [Card.SEVEN] * 4 + [Card.THREE],
            TurnPosition.THIRD: [Card.FIVE] * 4 + [Card.FOUR] * 4 + [Card.SIX] * 4 + [Card.TWO] * 4 + [Card.THREE] * 2 + [Card.LITTLE_JOKER] + [Card.BIG_JOKER]
        }
        game.betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 2)
        self.assertTrue(game.get_current_position() == TurnPosition.THIRD)
        game.play_move(None)
        self.assertTrue(game.get_current_position() == TurnPosition.THIRD.next())
        self.assertTrue(len(game.get_move_logs()) == 1)
        self.assertTrue(game.get_move_logs()[0][1] is None)
        game.play_move(SpecificMove(RankedMoveType(MoveType.BOMB, Card.KING), cards=Counter({Card.KING: 4})))
        self.assertTrue(game.get_current_position() == TurnPosition.SECOND)
        feature_matrix = players[1].derive_features(game)
        self.assertTrue(feature_matrix[0][-5] == 1)
        self.assertTrue(feature_matrix[0][-2] == 1)
        self.assertTrue(feature_matrix[1][10] == 4)
        self.assertTrue(np.sum(feature_matrix) == 7)

    def test_betting(self):
        players = [LearningPlayer_v1('v1', None)] * 3
        game = LandlordGame(players=players)
        game.force_current_position(TurnPosition.SECOND)
        game.force_kitty([Card.LITTLE_JOKER, Card.BIG_JOKER, Card.THREE])
        game.make_bet_move(BetMove(2))
        game.make_bet_move(None)
        game.make_bet_move(BetMove(3))
        game.reveal_kitty()
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4 + [Card.KING] * 4 + [Card.QUEEN] * 4 + [Card.JACK] * 4 + [Card.THREE],
            TurnPosition.SECOND: [Card.TEN] * 4 + [Card.NINE] * 4 + [Card.EIGHT] * 4 + [Card.SEVEN] * 4 + [Card.THREE],
            TurnPosition.THIRD: [Card.FIVE] * 4 + [Card.FOUR] * 4 + [Card.SIX] * 4 + [Card.TWO] * 4 + [Card.THREE] * 1
        }
        game.force_setup(TurnPosition.FIRST, hands, 2)
        game.play_move(SpecificMove(RankedMoveType(MoveType.BOMB, Card.ACE), cards=Counter({Card.ACE: 4})))
        feature_matrix = players[1].derive_features(game)
        self.assertTrue(feature_matrix[0][-3] == 2)
        self.assertTrue(np.sum(players[1].derive_features(game)) == 16)

if __name__ == '__main__':
    unittest.main()

