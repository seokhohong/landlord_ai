
import random

from src.game.landlord import LandlordGame


class Simulator:
    def __init__(self, rounds, player_pool):
        self.rounds = rounds
        self.player_pool = player_pool

    def play_game(self, num_rounds=1):
        players = self.pick_players()
        game = LandlordGame(players=players, num_rounds=num_rounds)
        game.play_rounds()


    def pick_players(self):
        return [random.choice(self.player_pool) for _ in range(LandlordGame.NUM_PLAYERS)]

