import unittest

from landlordai.game.player import LearningPlayer_v1
from landlordai.sim.simulate import Simulator

import numpy as np


class TestLandlordMethods(unittest.TestCase):
    def test_simulator(self):
        players = [LearningPlayer_v1(name='random') for _ in range(5)]
        simulator = Simulator(5, players)
        simulator.play_rounds()

        history_matrices, move_vectors, hand_vectors, qs = simulator.get_sparse_game_data()
        self.assertTrue(len(hand_vectors) == len(move_vectors))
        self.assertTrue(history_matrices[0].shape[0] == LearningPlayer_v1.TIMESTEPS)
        self.assertTrue(len(history_matrices) == qs.shape[0])
        self.assertTrue(len(move_vectors) == len(history_matrices))

        self.assertTrue(np.max(np.abs(qs)) < 1.5)

if __name__ == '__main__':
    unittest.main()

