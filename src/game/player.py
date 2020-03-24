from enum import Enum, IntEnum
from src.game.deck import CardSet, Card
from collections import Counter
import random
import numpy as np
import keras
from keras.layers import *

from src.game.move import KittyReveal, SpecificMove, BetMove


class TurnPosition(IntEnum):
    FIRST = 0,
    SECOND = 1,
    THIRD = 2

    def next(self):
        if self == TurnPosition.FIRST:
            return TurnPosition.SECOND
        if self == TurnPosition.SECOND:
            return TurnPosition.THIRD
        if self == TurnPosition.THIRD:
            return TurnPosition.FIRST

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.name + ' Player'

class Player:
    def __init__(self, name):
        self.name = name

    # returns None for pass
    def make_bet(self, game):
        return None

    # returns None for pass, otherwise returns a SpecificMove
    def make_move(self, game, player: TurnPosition):
        pass

class LearningPlayer_v1(Player):
    TIMESTEPS = 100
    # 12: number of distinct cards, each feature is the number played
    # 3: one-hot encoding for landlord player
    # 1: feature for points bet
    # 1: separate boolean for pass
    # 1: separate boolean for vault reveal
    TIMESTEP_FEATURES = len(Card) + 6

    def __init__(self, name, net_files):
        super().__init__(name)
        if net_files is None:
            self.create_nnet()
        else:
            self.load_nnets(net_files)

    def load_nnets(self, net_files):
        self.history_nnet = keras.models.load_model(net_files[0])
        self.position_nnet = keras.models.load_model(net_files[1])

    def create_nnet(self):
        # number of options we'll inference at once
        GRU_DIM = 64

        inp = Input((LearningPlayer_v1.TIMESTEPS, LearningPlayer_v1.TIMESTEP_FEATURES))
        gru = GRU(64)(inp)
        self.history_net = keras.models.Model(inputs=[inp], outputs=gru)
        self.history_net.compile(loss='mean_square_error', optimizer='adam', metrics=['mean_squared_error'])

        history_inp = Input((GRU_DIM))
        play_inp = Input((LearningPlayer_v1.TIMESTEP_FEATURES))
        dense_1 = Dense(32, activation='relu')(Concatenate([history_inp, play_inp]))
        dense_2 = Dense(1, activation='sigmoid')(dense_1)
        self.eval_net = keras.models.Model(inputs=[history_inp, play_inp], outputs=[dense_2])
        self.eval_net.compile(loss='mean_square_error', optimizer='adam', metrics = ['mean_squared_error'])

    @classmethod
    def feature_index(cls, i):
        'maps card to feature index'
        if cls.feature_mapping is None:
            feature_list = [[card.get_name() for card in Card]]
            feature_list.extend(['I_AM_LANDLORD',
                                 'I_AM_BEFORE_LANDLORD',
                                 'I_AM_AFTER_LANDLORD',
                                 'POINTS_BET',
                                 'IS_PASS',
                                'REVEALING_KITTY'])

            cls.feature_mapping = dict([(i, value) for value, i in enumerate(feature_list)])

        return cls.feature_mapping[i]

    def derive_features(self, game):
        feature_matrix = np.zeros((LearningPlayer_v1.TIMESTEPS, LearningPlayer_v1.TIMESTEP_FEATURES))
        for i, move in enumerate(game.get_game_logs()):
            other_features = {}
            player_position, step = move
            if type(step) == BetMove:
                other_features = {
                    'IS_PASS': 1
                }
            if step is None:
                other_features = {
                    'IS_PASS': 1
                }
            if type(step) == KittyReveal:
                for card, count in step.cards.items():
                    feature_matrix[i, LearningPlayer_v1.feature_index(card)] = count


            if type(step) == SpecificMove:
                for card, count in step.cards.items():
                    feature_matrix[i, LearningPlayer_v1.feature_index(card)] = count
                other_features = {
                    'I_AM_LANDLORD': 1 if game.current_player_is_landlord() else 0,
                    'I_AM_BEFORE_LANDLORD': 1 if game.get_current_position().before() == game.get_landlord_position() else 0,
                    'I_AM_AFTER_LANDLORD': 1 if game.get_current_position().before() == game.get_landlord_position() else 0,
                }
            for feature, value in other_features.items():
                feature_matrix[i, LearningPlayer_v1.feature_index(feature)] = value

class RandomPlayer(Player):
    def make_bet(self, game):
        if game.get_bet_amount() < game.MAX_BET:
            return game.get_bet_amount() + 1
        return None

    def make_move(self, game, player: TurnPosition):
        hand = CardSet(Counter(game.get_hand(player)))
        all_moves = hand.get_all_moves()
        if game.get_control_position() != player:
            legal_moves = [move for move in all_moves if move.beats(game.get_last_played())]
        else:
            # we have control
            legal_moves = all_moves

        if len(legal_moves) == 0:
            return None
        return random.choice(legal_moves)

class NoBetPlayer(Player):
    def make_bet(self, game):
        return None