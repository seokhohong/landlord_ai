import unittest

from landlordai.game.landlord import LandlordGame
from landlordai.game.player import LearningPlayer
from landlordai.sim.simulate import Simulator, NoRecordsException

import numpy as np


class TestLandlordMethods(unittest.TestCase):

    def test_simulator_actual_q(self):
        players = [LearningPlayer(name='random', estimation_mode=LearningPlayer.ACTUAL_Q) for _ in range(5)]
        simulator = Simulator(2, players)
        simulator.play_rounds()

        history_matrices, move_vectors, hand_vectors, qs = simulator.get_game_data()
        self.assertTrue(len(hand_vectors) == len(move_vectors))
        self.assertTrue(history_matrices[0].shape[0] == LearningPlayer.TIMESTEPS)
        self.assertTrue(len(history_matrices) == qs.shape[0])
        self.assertTrue(len(move_vectors) == len(history_matrices))

    def test_competitor_pool(self):
        players = [LearningPlayer(name='random', estimation_mode=LearningPlayer.NO_ESTIMATION)]
        competitors = [LearningPlayer(name='random', estimation_mode=LearningPlayer.ACTUAL_Q) for _ in range(2)]
        simulator = Simulator(2, players, competitors)
        simulator.play_rounds()
        self.assertRaises(NoRecordsException, simulator.get_game_data)


if __name__ == '__main__':
    unittest.main()

