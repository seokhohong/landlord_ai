import numpy as np

from landlordai.game.landlord import LandlordGame
from landlordai.game.player import TurnPosition
import math

class GameStats:
    ELO_K = 10
    def __init__(self, player_pool, game_record):
        self.player_pool = player_pool

        unique_player_names = set([player.get_name() for player in player_pool])
        self.win_matrix = np.zeros((len(unique_player_names), len(unique_player_names)))
        self.loss_matrix = np.zeros((len(unique_player_names), len(unique_player_names)))
        self.player_map = dict([(player, i) for (i, player) in enumerate(unique_player_names)])

        self.elos = [1500] * len(unique_player_names)
        self.process_stats(game_record)

    def process_stats(self, game_record):
        for winner, loser in game_record:
            self.win_matrix[self.player_map[winner], self.player_map[loser]] += 1
            self.loss_matrix[self.player_map[loser], self.player_map[winner]] += 1
            self.process_elo(winner, loser)

    @classmethod
    def elo_expected(cls, a, b):
        return 1. / (1 + math.pow(10, (b - a) / 400))

    def recenter_elo(self):
        diff = 1500 - np.mean(self.elos)
        self.elos = [elo + diff / len(self.elos) for elo in self.elos]

    def process_elo(self, winner: str, loser: str):
        elo_winner = self.elos[self.player_map[winner]]
        elo_loser = self.elos[self.player_map[loser]]

        winner_expected = GameStats.elo_expected(elo_winner, elo_loser)
        loser_expected = GameStats.elo_expected(elo_loser, elo_winner)

        self.elos[self.player_map[winner]] += GameStats.ELO_K * (1 - winner_expected)
        self.elos[self.player_map[loser]] += GameStats.ELO_K * (0 - loser_expected)

        self.recenter_elo()

    def get_elo(self, player_name: str):
        return self.elos[self.player_map[player_name]]

    def get_win_rate(self, player_name: str):
        player_index = self.player_map[player_name]
        wins = np.sum(self.win_matrix[player_index]) - self.win_matrix[player_index, player_index]
        losses = np.sum(self.loss_matrix[player_index]) - self.loss_matrix[player_index, player_index]

        total_games = wins + losses
        return wins / total_games