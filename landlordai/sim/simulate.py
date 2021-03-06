
import random

import numpy as np
from scipy import sparse

from landlordai.game.landlord import LandlordGame
from landlordai.game.player import TurnPosition
from copy import copy

from tqdm import tqdm


class Simulator:
    # competitors are not used for feature extraction
    def __init__(self, rounds, player_pool, competitor_pool=None, record_loser_pct=0.1):
        if competitor_pool is None:
            competitor_pool = []
        self.rounds = rounds
        self.player_pool = player_pool + competitor_pool
        self.competitor_pool = competitor_pool
        self.record_everyone_pct = record_loser_pct
        self.record_states = []
        self.move_vectors = []
        self.hand_vectors = []
        self.q = []

        self.results = []

    def play_rounds(self, debug=False):
        for r in tqdm(range(self.rounds)):
            if debug:
                print('Playing Round ', r)
            self.play_game()
        if debug:
            print('Done Playing')

    def record_player(self, game, player):
        # don't record any competitors
        if player in self.competitor_pool:
            return

        player.compute_future_q(game)
        self.record_states.extend(player.get_record_history_matrices())
        self.move_vectors.extend(player.get_record_move_vectors())
        self.hand_vectors.extend(player.get_record_hand_vectors())
        self.q.append(player.get_estimated_qs())
        player._reset_records()

    def play_game(self):
        while True:
            players = self.pick_players()
            game = LandlordGame(players=players)
            # play a meaningful game
            game.play_round()
            if game.has_winners():
                players_to_record = game._winners
                if random.random() < self.record_everyone_pct:
                    players_to_record = list(TurnPosition)
                for pos in players_to_record:
                    player = game.get_ai(pos)
                    self.record_player(game, player)
                self.track_stats(game)
                break

            # clear out in case a full game wasn't played
            for player in players:
                player._reset_records()

    def track_stats(self, game: LandlordGame):
        assert game.is_round_over()
        winners = tuple([player.get_name() for player in game.get_winner_ais()])
        losers = tuple([player.get_name() for player in game.get_loser_ais()])
        self.results.append((winners, losers))

    def pick_players(self):
        return random.sample(self.player_pool, LandlordGame.NUM_PLAYERS)

    def get_game_data(self):
        if len(self.record_states) == 0:
            raise NoRecordsException
        return self.record_states, np.vstack(self.move_vectors), np.vstack(self.hand_vectors), np.hstack(self.q)

    def get_results(self):
        return copy(self.results)


class NoRecordsException(Exception):
    pass