from landlord.game.card import Card
from landlord.game.landlord import LandlordGame
from landlord.game.move import SpecificMove, RankedMoveType, MoveType, BetMove
from landlord.game.player import RandomPlayer, NoBetPlayer, LearningPlayer_v1, TurnPosition
from collections import Counter
import numpy as np
import unittest
from copy import copy

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
        self.assertTrue(np.abs(players[0].record_future_q[-1]) > 0.5)

if __name__ == '__main__':
    unittest.main()

