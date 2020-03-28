from src.game.card import Card
from src.game.landlord import LandlordGame
from src.game.move import SpecificMove, RankedMoveType, MoveType, BetMove
from src.game.player import RandomPlayer, NoBetPlayer, LearningPlayer_v1, TurnPosition
from collections import Counter
import numpy as np
import unittest
from copy import copy

from src.sim.simulate import Simulator


class TestLandlordMethods(unittest.TestCase):
    def test_simulator_move(self):
        players = [LearningPlayer_v1(name='random') for i in range(5)]
        simulator = Simulator(10, players)
        simulator.play_rounds()

        matrices, qs = simulator.get_sparse_game_data()
        self.assertTrue(matrices[0].shape[0] == LearningPlayer_v1.TIMESTEPS)
        self.assertTrue(len(matrices) == qs.shape[0])

if __name__ == '__main__':
    unittest.main()

