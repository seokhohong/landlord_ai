import unittest

from landlordai.game.landlord import LandlordGame
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

        #self.assertTrue(np.max(np.abs(qs)) < 1.5)

    def test_densify_matrix(self):
        players = [LearningPlayer_v1(name='random') for _ in range(5)]
        simulator = Simulator(1, players)
        simulator.play_rounds()

        history_matrices, _, _, _ = simulator.get_sparse_game_data()

        def naive_densify(history_matrices):
            return np.array([x.todense() for x in history_matrices])

        def accel_densify(history_matrices):
            assert len(history_matrices) > 0
            mat = np.zeros((len(history_matrices), history_matrices[0].shape[0], history_matrices[0].shape[1]))
            for i, matrix in enumerate(history_matrices):
                nonzero = matrix.nonzero()
                for x, y in zip(nonzero[0], nonzero[1]):
                    mat[i, x, y] = matrix[x, y]
            return mat

        self.assertTrue(np.allclose(naive_densify(history_matrices), accel_densify(history_matrices)))

if __name__ == '__main__':
    unittest.main()

