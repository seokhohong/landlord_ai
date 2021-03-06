import unittest
from collections import Counter
from copy import copy

import numpy as np

from landlordai.game.card import Card
from landlordai.game.landlord import LandlordGame
from landlordai.game.move import SpecificMove, RankedMoveType, MoveType, BetMove
from landlordai.game.player import RandomPlayer, NoBetPlayer, LearningPlayer, TurnPosition


class TestLandlordMethods(unittest.TestCase):

    def test_extended_game(self):
        game = LandlordGame(players=[RandomPlayer(name='random')] * 3)
        game.play_round()
        game2 = copy(game)
        self.assertTrue(game.get_move_logs() == game2.get_move_logs())
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
        self.assertTrue(np.sum(np.abs(game._scores)) == 0)

    def test_setup(self):
        players = [LearningPlayer('v1', None)] * 3
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4 + [Card.KING] * 4 + [Card.QUEEN] * 4 + [Card.JACK] * 4 + [Card.THREE],
            TurnPosition.SECOND: [Card.TEN] * 4 + [Card.NINE] * 4 + [Card.EIGHT] * 4 + [Card.SEVEN] * 4 + [Card.THREE],
            TurnPosition.THIRD: [Card.FIVE] * 4 + [Card.FOUR] * 4 + [Card.SIX] * 4 + [Card.TWO] * 4 + [Card.THREE] * 2 + [Card.LITTLE_JOKER] + [Card.BIG_JOKER]
        }
        game._betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 2)
        self.assertTrue(game.get_current_position() == TurnPosition.THIRD)
        game.play_move(None)
        self.assertTrue(game.get_current_position() == TurnPosition.THIRD.next())
        self.assertTrue(len(game.get_move_logs()) == 1)
        self.assertTrue(game.get_move_logs()[0][1] is None)
        game.play_move(SpecificMove(RankedMoveType(MoveType.BOMB, Card.KING), cards=Counter({Card.KING: 4})))
        self.assertTrue(game.get_current_position() == TurnPosition.SECOND)
        feature_matrix = players[1]._derive_features(game)
        self.assertTrue(feature_matrix[0][-6] == 1)
        self.assertTrue(feature_matrix[0][-2] == 1)
        self.assertTrue(feature_matrix[1][10] == 4)
        self.assertTrue(np.sum(feature_matrix) == 7)

    def test_betting(self):
        players = [LearningPlayer('v1')] * 3
        game = LandlordGame(players=players)
        game.force_current_position(TurnPosition.SECOND)
        game.force_kitty([Card.LITTLE_JOKER, Card.BIG_JOKER, Card.THREE])
        game._make_bet_move(BetMove(2))
        game._make_bet_move(None)
        self.assertEqual(game.get_last_played(), BetMove(2))
        game._make_bet_move(BetMove(3))
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4 + [Card.KING] * 4 + [Card.QUEEN] * 4 + [Card.JACK] * 4 + [Card.THREE],
            TurnPosition.SECOND: [Card.TEN] * 4 + [Card.NINE] * 4 + [Card.EIGHT] * 4 + [Card.SEVEN] * 4 + [Card.THREE],
            TurnPosition.THIRD: [Card.FIVE] * 4 + [Card.FOUR] * 4 + [Card.SIX] * 4 + [Card.TWO] * 4 + [Card.THREE] * 1
        }
        game.force_setup(TurnPosition.FIRST, hands, 2)
        game.play_move(SpecificMove(RankedMoveType(MoveType.BOMB, Card.ACE), cards=Counter({Card.ACE: 4})))
        feature_matrix = players[1]._derive_features(game)
        self.assertTrue(feature_matrix[0][-3] == 2)
        self.assertTrue(np.sum(players[1]._derive_features(game)) == 16)

    def test_landlord_game_ending(self):
        players = [LearningPlayer('v1')] * 3
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4,
            TurnPosition.SECOND: [Card.TEN] * 4,
            TurnPosition.THIRD: [Card.FIVE] * 4
        }
        game._betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 2)
        self.assertTrue(game.move_ends_game(SpecificMove(RankedMoveType(MoveType.BOMB, Card.FIVE), Counter({Card.FIVE: 4}))))
        self.assertFalse(game.move_ends_game(SpecificMove(RankedMoveType(MoveType.BOMB, Card.TEN), Counter({Card.TEN: 4}))))

    def test_endgame_scenario(self):
        players = [LearningPlayer('v1')] * 3
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 1,
            TurnPosition.SECOND: [Card.TEN] * 1,
            TurnPosition.THIRD: [Card.JACK, Card.QUEEN]
        }
        game._betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 2)
        game.main_game()
        self.assertTrue(TurnPosition.FIRST in game.get_winners())
        self.assertTrue(len(game.get_move_logs()) == 2)

    def test_landlord_bombing(self):
        players = [LearningPlayer('v1')] * 3
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4 + [Card.THREE],
            TurnPosition.SECOND: [Card.TEN] * 4 + [Card.THREE],
            TurnPosition.THIRD: [Card.FIVE] * 4 + [Card.THREE]
        }
        game._betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 2)
        game.play_move(SpecificMove(RankedMoveType(MoveType.BOMB, Card.FIVE), Counter({Card.FIVE: 4})))
        self.assertTrue(game._bet_amount == 4)

    def test_bet_1(self):
        players = [LearningPlayer('v1')] * 3
        game = LandlordGame(players=players)
        game.play_move(BetMove(0))
        game.play_move(BetMove(0))
        self.assertFalse(game.is_round_over())
        self.assertFalse(game.is_betting_complete())
        self.assertFalse(game.move_ends_game(BetMove(1)))
        self.assertTrue(game.move_ends_game(BetMove(0)))
        game.play_move(BetMove(0))
        self.assertTrue(game.is_round_over())
        self.assertTrue(game.is_betting_complete())

    def test_bet_2(self):
        players = [LearningPlayer('v1')] * 3
        game = LandlordGame(players=players)
        game.play_move(BetMove(0))
        self.assertEqual(len(game.get_legal_moves()), 4)
        one_better = game.get_current_position()
        game.play_move(BetMove(1))
        self.assertFalse(game.is_round_over())
        self.assertFalse(game.is_betting_complete())
        game.play_move(BetMove(0))
        self.assertFalse(game.is_round_over())
        self.assertTrue(game.is_betting_complete())
        self.assertTrue(game.get_bet_amount() == 1)
        self.assertEqual(game.get_current_position(), one_better)

    def test_bet_3(self):
        players = [LearningPlayer('v1')] * 3
        game = LandlordGame(players=players)
        game.play_move(BetMove(0))
        game.play_move(BetMove(3))
        self.assertFalse(game.is_round_over())
        self.assertTrue(game.is_betting_complete())
        self.assertTrue(game.get_bet_amount() == 3)

    def test_sweep(self):
        players = [LearningPlayer('v1')] * 3
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * LandlordGame.DEAL_SIZE,
            TurnPosition.SECOND: [Card.TEN] * LandlordGame.DEAL_SIZE,
            TurnPosition.THIRD: [Card.FIVE] * 4
        }
        game._betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 2)
        game.play_move(SpecificMove(RankedMoveType(MoveType.BOMB, Card.FIVE), Counter({Card.FIVE: 4})))
        self.assertTrue(game.peasants_have_no_plays())
        self.assertTrue(game.get_scores()[TurnPosition.THIRD] == 2 * 2 * 2 * LandlordGame.SWEEP_MULTIPLIER)
        self.assertEqual(game.get_r(), 24)
        self.assertEqual(game.get_winbased_r(), 1)

if __name__ == '__main__':
    unittest.main()

