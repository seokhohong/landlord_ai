from src.game.deck import Card
from src.game.landlord import LandlordGame
from src.game.move import SpecificMove, RankedMoveType, MoveType
from src.game.player import RandomPlayer, NoBetPlayer, LearningPlayer_v1, TurnPosition
from collections import Counter
import numpy as np
import unittest

class TestLandlordMethods(unittest.TestCase):
    def test_basic_game(self):
        game = LandlordGame(players=[RandomPlayer(name='random')] * 3)
        game.play_round()

    def test_extended_game(self):
        game = LandlordGame(players=[RandomPlayer(name='random')] * 3)
        game.play_rounds()
        for line in game.string_logs:
            print(line)

    def test_nobet_game(self):
        game = LandlordGame(players=[NoBetPlayer(name='random')] * 3)
        game.play_rounds()
        self.assertTrue(np.sum(np.abs(game.scores)) == 0)

    def test_setup(self):
        players = [LearningPlayer_v1('v1', None)] * 3
        game = LandlordGame(players=players)
        hands = [
            [Card.ACE] * 4 + [Card.KING] * 4 + [Card.QUEEN] * 4 + [Card.JACK] * 4 + [Card.THREE],
            [Card.TEN] * 4 + [Card.NINE] * 4 + [Card.EIGHT] * 4 + [Card.SEVEN] * 4 + [Card.THREE],
            [Card.FIVE] * 4 + [Card.FOUR] * 4 + [Card.SIX] * 4 + [Card.TWO] * 4 + [Card.THREE] * 2 + [Card.LITTLE_JOKER] + Card.BIG_JOKER,
        ]
        game.force_setup(TurnPosition.THIRD, hands)
        game.play_move(None)
        self.assertTrue(len(game.get_move_logs()) == 1)
        self.assertTrue(game.get_move_logs()[0] is None)
        game.play_move(SpecificMove(RankedMoveType(MoveType.BOMB, Card.KING), cards=Counter({Card.KING: 4})))


if __name__ == '__main__':
    unittest.main()

