import random
from collections import Counter
from copy import copy
from enum import IntEnum

import keras
import numpy as np
from keras.layers import *
from keras.losses import mean_squared_error
from keras import backend as K

from landlordai.game.card import Card
from landlordai.game.deck import CardSet
from landlordai.game.move import KittyReveal, SpecificMove, BetMove


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

    def previous(self):
        return self.next().next()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.name + ' Player'

class Player:
    def __init__(self, name):
        self.name = name

    # returns None for pass
    def make_bet(self, game, player: TurnPosition):
        return None

    # returns None for pass, otherwise returns a SpecificMove
    def make_move(self, game, player: TurnPosition):
        pass

class LearningPlayer_v1(Player):
    TIMESTEPS = 100
    # 12: number of distinct cards, each feature is the number played
    # 3: one-hot encoding for landlordai player
    # 1: feature for points bet
    # 1: separate boolean for pass
    # 1: separate boolean for vault reveal
    TIMESTEP_FEATURES = len(Card) + 6

    def __init__(self, name, epsilon=0.1, net_files=None):
        super().__init__(name)
        if net_files is None:
            self.create_nnet()
        else:
            self.load_nnets(net_files)

        self.epsilon = 0.1
        self.feature_index = self.make_feature_index()

        # elements for record
        self.reset_records()

    def reset_records(self):
        self.record_history_matrices = []
        self.record_move_vectors = []
        self.record_future_q = []

    def load_nnets(self, net_files):
        self.history_net = keras.models.load_model(net_files[0])
        self.position_net = keras.models.load_model(net_files[1])

    def create_nnet(self):
        # number of options we'll inference at once
        GRU_DIM = 64

        inp = Input((LearningPlayer_v1.TIMESTEPS, LearningPlayer_v1.TIMESTEP_FEATURES))
        gru = GRU(GRU_DIM)(inp)

        self.history_net = keras.models.Model(inputs=[inp], outputs=gru)
        self.history_net.compile(loss=mean_squared_error, optimizer='adam', metrics=['mean_squared_error'])

        history_inp = Input((GRU_DIM, ))
        play_inp = Input((LearningPlayer_v1.TIMESTEP_FEATURES, ))
        dense_1 = Dense(32, activation='relu')(Concatenate()([history_inp, play_inp]))
        dense_2 = Dense(1, activation='linear')(dense_1)
        self.position_net = keras.models.Model(inputs=[history_inp, play_inp], outputs=[dense_2])
        self.position_net.compile(loss=mean_squared_error, optimizer='adam', metrics = ['mean_squared_error'])

    def make_feature_index(self):
        'maps card to feature index'
        feature_list = [card.get_name() for card in Card]
        feature_list.extend(['I_AM_LANDLORD',
                             'I_AM_BEFORE_LANDLORD',
                             'I_AM_AFTER_LANDLORD',
                             'POINTS_BET',
                             'IS_PASS',
                            'REVEALING_KITTY'])

        return dict([(i, value) for value, i in enumerate(feature_list)])

    def get_feature_index(self, i):
        return self.feature_index[i]

    def _derive_move_stack(self, game):
        return np.vstack([self.compute_move_vector(player, game.get_landlord_position(), move) for player, move in game.get_game_logs()])

    def _append_padding(self, move_stack):
        fluff_volume = LearningPlayer_v1.TIMESTEPS - len(move_stack)

        assert fluff_volume >= 0
        fluff_stack = np.zeros((fluff_volume, LearningPlayer_v1.TIMESTEP_FEATURES))

        if len(move_stack) == 0:
            return fluff_stack

        return np.vstack([move_stack, fluff_stack])

    def derive_features(self, game):
        if len(game.get_game_logs()) <= 0:
            move_stack = []
        else:
            move_stack = self._derive_move_stack(game)
        return self._append_padding(move_stack)

    # separately returns current game state - 1 = history_matrix, and separate move vector
    def derive_record_features(self, game):
        move_vector = np.zeros(LearningPlayer_v1.TIMESTEP_FEATURES)
        if len(game.get_game_logs()) == 1:
            last_player, last_move = game.get_game_logs()[0]
            move_vector = self.compute_move_vector(last_player, game.get_landlord_position(), last_move)
        if len(game.get_game_logs()) <= 1:
            history_matrix = self._append_padding([])
            return history_matrix, move_vector

        move_stack = self._derive_move_stack(game)
        return self._append_padding(move_stack[:-1]), move_stack[-1]

    def compute_move_vector(self, player: TurnPosition, landlord_position: TurnPosition, move):
        move_vector = np.zeros(LearningPlayer_v1.TIMESTEP_FEATURES)
        other_features = {}
        if type(move) == BetMove:
            other_features = {
                'POINTS_BET': move.get_amount()
            }
        if move is None:
            other_features = {'IS_PASS': 1,
                              'I_AM_LANDLORD': 1 if player == landlord_position else 0,
                              'I_AM_BEFORE_LANDLORD': 1 if player.previous() == landlord_position else 0,
                              'I_AM_AFTER_LANDLORD': 1 if player.next() == landlord_position else 0}

        if type(move) == KittyReveal:
            for card in move.cards:
                move_vector[self.get_feature_index(card.name)] += 1
            other_features = {
                'REVEALING_KITTY': 1
            }

        if type(move) == SpecificMove:
            for card, count in move.cards.items():
                move_vector[self.get_feature_index(card.name)] = count
            # print(1 if game.current_player_is_landlord() else 0)
            # print(1 if game.get_current_position().next() == game.get_landlord_position() else 0)
            # print(1 if game.get_current_position().previous() == game.get_landlord_position() else 0)
            other_features = {'I_AM_LANDLORD': 1 if player == landlord_position else 0,
                              'I_AM_BEFORE_LANDLORD': 1 if player.previous() == landlord_position else 0,
                              'I_AM_AFTER_LANDLORD': 1 if player.next() == landlord_position else 0}
        for feature, value in other_features.items():
            move_vector[self.get_feature_index(feature)] = value

        return move_vector

    # hard_clamp means we only consider win states to have value (useful when the network is not initialized)
    def compute_state_value(self, game, has_won, hard_clamp=False):
        if has_won:
            return 1
        if hard_clamp:
            return 0
        return self.history_net.predict(self.derive_features(game))

    def get_history_vector(self, features):
        return self.history_net.predict(np.array([features]))[0]

    def decide_best_move(self, game, player: TurnPosition):
        assert len(game.get_hand(player)) > 0
        legal_moves = game.get_legal_moves(player)

        history_matrix = self.derive_features(game)

        # all the moves we make from here will not affect the history, so assess it and copy
        history_vector = self.get_history_vector(history_matrix)
        history_matrix = np.tile(history_vector, (len(legal_moves), 1))

        # create features for each of the possible moves from this position
        move_options_matrix = np.vstack([self.compute_move_vector(game.get_current_position(), game.get_landlord_position(), move) for move in legal_moves])

        predictions = self.position_net.predict([history_matrix, move_options_matrix]).reshape(len(legal_moves))

        best_move_index = np.argmax(predictions)
        best_move = legal_moves[best_move_index]
        best_move_q = predictions[best_move_index]

        return best_move, best_move_q

    def make_bet(self, game, player: TurnPosition):
        return self.make_move(game, player)

    def make_move(self, game, player: TurnPosition):
        best_move, _ = self.decide_best_move(game, player)

        self.record_move(game, best_move, player)

        return best_move

    def record_move(self, game, best_move, player: TurnPosition):
        previous_history_matrix, move_vector = self.derive_record_features(game)

        copy_game = copy(game)
        copy_game.play_move(best_move)
        if copy_game.is_round_over():
            future_reward = copy_game.get_r()
        else:
            best_next_move, best_next_move_q = self.decide_best_move(copy_game, player.next())
            copy_game.play_move(best_next_move)
            if copy_game.is_round_over():
                future_reward = copy_game.get_r()
            else:
                future_reward = best_next_move_q + copy_game.get_r()

        self.record_history_matrices.append(previous_history_matrix)
        self.record_move_vectors.append(move_vector)
        self.record_future_q.append(future_reward)

    def get_record_history_matrices(self):
        return self.record_history_matrices

    def get_record_move_vectors(self):
        return self.record_move_vectors

    def get_future_q(self):
        return self.record_future_q


class RandomPlayer(Player):
    def make_bet(self, game, player: TurnPosition):
        if game.get_bet_amount() < game.MAX_BET:
            return BetMove(game.get_bet_amount() + 1)
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
    def make_bet(self, game, player: TurnPosition):
        return None