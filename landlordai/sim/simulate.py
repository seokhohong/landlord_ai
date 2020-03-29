
import random

import numpy as np
from scipy import sparse

from landlordai.game.landlord import LandlordGame


class Simulator:
    def __init__(self, rounds, player_pool):
        self.rounds = rounds
        self.player_pool = player_pool
        self.sparse_record_states = []
        self.move_vectors = []
        self.q = []

    def play_rounds(self, debug=False):
        for r in range(self.rounds):
            if debug:
                print('Playing Round ', r)
            self.play_game()
        if debug:
            print('Done Playing')

    def play_game(self):
        players = self.pick_players()
        game = LandlordGame(players=players)
        # play a meaningful game
        while True:
            game.play_round()
            if game.has_winners():
                for pos in game.winners:
                    player = game.get_ai(pos)
                    self.sparse_record_states.extend([sparse.csr_matrix(x) for x in player.get_record_history_matrices()])
                    self.move_vectors.extend(player.get_record_move_vectors())
                    self.q.append(player.get_future_q())
                    player.reset_records()
                break

    def pick_players(self):
        return random.sample(self.player_pool, LandlordGame.NUM_PLAYERS)

    def get_sparse_game_data(self):
        return self.sparse_record_states, np.vstack(self.move_vectors), np.hstack(self.q)
