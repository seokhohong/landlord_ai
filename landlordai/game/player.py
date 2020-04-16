import random
from collections import Counter
from copy import copy
from enum import IntEnum

import keras
import numpy as np

from landlordai.game.card import Card, string_to_card
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

    # returns None for pass, otherwise returns a SpecificMove
    def make_move(self, game):
        pass

    def get_name(self):
        return self.name


class LearningPlayer(Player):
    TIMESTEPS = 100

    #estimation methods for q(s', a')
    NO_ESTIMATION = 'no_estimation'
    ACTUAL_Q = 'actualq'

    # 12: number of distinct cards, each feature is the number played
    # 3: one-hot encoding for landlordai player
    # 1: feature for points bet
    # 1: separate boolean for pass
    # 1: separate boolean for vault reveal
    TIMESTEP_FEATURES = len(Card) + 6
    # appends the length of each player's hand
    HAND_FEATURES = len(Card) + 3

    def __init__(self, name, net_dir=None, epsilon=0.1, learning_rate=0.2, discount_factor=1,
                  estimation_mode=ACTUAL_Q,
                 random_mc_num_explorations=30,
                 estimation_depth=1):
        super().__init__(name)

        self.epsilon = epsilon
        self.empty_nets = False
        self.estimation_mode = estimation_mode
        self.random_mc_num_explorations = random_mc_num_explorations
        self.estimation_depth = estimation_depth
        if net_dir is None:
            self.empty_nets = True
        else:
            self.load_nnets(net_dir)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.feature_index = self.make_feature_index()

        # elements for record
        self.reset_records()

    def reset_records(self):
        self.record_history_matrices = []
        self.record_move_vectors = []
        self._record_future_q = []
        self.record_hand_vectors = []
        # for debugging
        self._record_state_q = []
        self._recording_finalized = False

    def load_nnets(self, net_dir):
        self.history_net = keras.models.load_model(net_dir + "/history.h5")
        self.position_net = keras.models.load_model(net_dir + '/position.h5')

    '''
    def create_nnet(self):
        # number of options we'll inference at once
        GRU_DIM = 64

        inp = Input((LearningPlayer_v1.TIMESTEPS, LearningPlayer_v1.TIMESTEP_FEATURES))
        gru = GRU(GRU_DIM)(inp)

        self.history_net = keras.models.Model(inputs=[inp], outputs=gru)
        self.history_net.compile(loss=mean_squared_error, optimizer='adam', metrics=['mean_squared_error'])

        history_inp = Input((GRU_DIM, ))
        play_inp = Input((LearningPlayer_v1.TIMESTEP_FEATURES, ))
        hand_inp = Input((LearningPlayer_v1.HAND_FEATURES, ))

        dense_1 = Dense(32, activation='relu')(Concatenate()([history_inp, play_inp, hand_inp]))
        dense_2 = Dense(1, activation='linear')(dense_1)
        self.position_net = keras.models.Model(inputs=[history_inp, play_inp, hand_inp], outputs=[dense_2])
        self.position_net.compile(loss=mean_squared_error, optimizer='adam', metrics=['mean_squared_error'])

    '''

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
        # pad to achieve timestep length
        feature_stack = []
        move_stack = [self.compute_move_vector(player, game.get_landlord_position(), move) for player, move in game.get_game_logs()]
        feature_stack.extend(move_stack)
        if len(feature_stack) == 0:
            return []
        return np.vstack(feature_stack)

    def _derive_move_stack_bridge(self, game, player):
        # pad to achieve timestep length
        feature_stack = [np.concatenate([self.get_hand_vector(game, player),
                                         np.zeros(LearningPlayer.TIMESTEP_FEATURES - LearningPlayer.HAND_FEATURES)])]
        move_stack = [self.compute_move_vector(player, game.get_landlord_position(), move) for player, move in game.get_game_logs()]
        feature_stack.extend(move_stack)
        return np.vstack(feature_stack)

    def _append_padding(self, move_stack):
        fluff_volume = LearningPlayer.TIMESTEPS - len(move_stack)

        assert fluff_volume >= 0
        fluff_stack = np.zeros((fluff_volume, LearningPlayer.TIMESTEP_FEATURES))

        if len(move_stack) == 0:
            return fluff_stack

        return np.vstack([move_stack, fluff_stack])

    def derive_features(self, game):
        return self._append_padding(self._derive_move_stack(game))

    def derive_features_bridge(self, game, player: TurnPosition):
        return self._append_padding(self._derive_move_stack_bridge(game, player))

    def compute_move_vector(self, player: TurnPosition, landlord_position: TurnPosition, move):
        move_vector = np.zeros(LearningPlayer.TIMESTEP_FEATURES)
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
            other_features = {'I_AM_LANDLORD': 1 if player == landlord_position else 0,
                              'I_AM_BEFORE_LANDLORD': 1 if player.previous() == landlord_position else 0,
                              'I_AM_AFTER_LANDLORD': 1 if player.next() == landlord_position else 0}
        for feature, value in other_features.items():
            move_vector[self.get_feature_index(feature)] = value

        return move_vector

    def get_hand_vector(self, game, player: TurnPosition):
        hand = game.get_hand(player)
        vector = np.zeros(len(Card) + 3)
        for i, card in enumerate(Card):
            vector[i] = hand.count(card)

        if game.is_betting_complete():
            vector[-3] = len(game.get_hand(game.get_landlord_position()))
            vector[-2] = len(game.get_hand(game.get_landlord_position().previous()))
            vector[-1] = len(game.get_hand(game.get_landlord_position().next()))
        else:
            vector[-3] = 17
            vector[-2] = 17
            vector[-1] = 17
        return vector

    def get_history_vector(self, features):
        if self.empty_nets:
            return np.random.random(LearningPlayer.TIMESTEP_FEATURES) * 0.01

        return self.history_net.predict(np.array([features]), batch_size=1)[0]

    def get_position_predictions(self, history_matrix, move_options_matrix, hand_matrix):
        if self.empty_nets:
            return np.random.random((move_options_matrix.shape[0])) * 0.01

        num_rows = move_options_matrix.shape[0]
        return self.position_net.predict([history_matrix, move_options_matrix, hand_matrix], batch_size=num_rows).reshape(num_rows)

    def make_bet_decision(self, game, legal_moves, predictions):
        bet_indices = []
        # issue with recursive import, so not using constant
        flip_predictions = np.copy(predictions)
        for i in range(3):
            bet_index = legal_moves.index([move for move in legal_moves if move.get_amount() == i][0])
            if i <= game.get_bet_amount():
                flip_predictions[bet_index] = - flip_predictions[bet_index]

        return np.argmax(flip_predictions)

    def full_move_evaluation(self, game, legal_moves):
        history_features = self.derive_features(game)

        # all the moves we make from here will not affect the history, so assess it and copy
        history_vector = self.get_history_vector(history_features)
        history_matrix = np.tile(history_vector, (len(legal_moves), 1))

        # make the hand vector and copy it
        hand_vector = self.get_hand_vector(game, game.get_current_position())
        hand_matrix = np.tile(hand_vector, (len(legal_moves), 1))

        # create features for each of the possible moves from this position
        move_options_matrix = np.vstack([self.compute_move_vector(game.get_current_position(),
                                                                  game.get_landlord_position(), move) for move in legal_moves])

        predictions = self.get_position_predictions(history_matrix, move_options_matrix, hand_matrix)

        return predictions

    def decide_best_move(self, game, debug=False):
        assert len(game.get_hand(game.get_current_position())) > 0
        legal_moves = game.get_legal_moves()

        predictions = self.full_move_evaluation(game, legal_moves)

        # for debugging
        raw_predictions = np.copy(predictions)

        # if the move ends the game, then force-score the position
        has_game_ending = False
        for i, move in enumerate(legal_moves):
            if game.move_ends_game(move):
                copy_game = copy(game)
                copy_game.play_move(move)
                predictions[i] = self.get_game_result(copy_game)
                has_game_ending = True

        best_move_index = 0
        if game.get_current_position() == game.get_landlord_position():
            best_move_index = np.argmax(predictions)
        elif game.is_betting_complete():
            best_move_index = np.argmin(predictions)
        else:
            # go landlord if it's worth it, otherwise peasant
            best_move_index = self.make_bet_decision(game, legal_moves, predictions)

        if debug:
            print('(' +  game.get_position_role_name(game.get_current_position()) + ') Player', self.get_name())
            print('Hand', game.get_hand(game.get_current_position()))
            for move, raw_pred, score in sorted(list(zip(legal_moves, raw_predictions, predictions)), key=lambda x : x[1]):
                if raw_pred != score:
                    print(raw_pred, score, move)
                else:
                    print(raw_pred, move)
            print('Made Move', legal_moves[best_move_index])

        if random.random() < self.epsilon and not has_game_ending:
            best_move_index = random.randint(0, len(predictions) - 1)
            if debug:
                print('Randomed', legal_moves[best_move_index])

        best_move = legal_moves[best_move_index]
        # return the predicted Q here, not the clamped Q
        best_move_q = raw_predictions[best_move_index]

        return best_move, best_move_q

    def make_move(self, game, debug=False):
        best_move, best_move_q = self.decide_best_move(game, debug=debug)

        if self.estimation_mode == LearningPlayer.ACTUAL_Q:
            self.record_move(game, best_move, best_move_q, game.get_current_position())

        return best_move

    def record_move(self, game, best_move, best_move_q, player: TurnPosition):
        history_matrix = self.derive_features(game)
        move_vector = self.compute_move_vector(player, game.get_landlord_position(), best_move)
        hand_vector = self.get_hand_vector(game, player)

        self.record_history_matrices.append(history_matrix)
        self.record_move_vectors.append(move_vector)
        self.record_hand_vectors.append(hand_vector)

        self._record_state_q.append(best_move_q)

    def compute_future_q(self, game):
        assert self._recording_finalized is False

        self._record_state_q.append(self.get_game_result(game))
        # shift forward
        self._record_future_q = []
        for i, experienced_q in enumerate(self._record_state_q[1:]):
            old_q = self._record_state_q[i]
            update_q = old_q + self.learning_rate * (self.discount_factor * experienced_q - old_q)
            self._record_future_q.append(update_q)

        self._recording_finalized = True

    def get_game_result(self, game):
        return game.get_r()

    def get_record_history_matrices(self):
        return self.record_history_matrices

    def get_record_move_vectors(self):
        return self.record_move_vectors

    def get_record_hand_vectors(self):
        return self.record_hand_vectors

    def get_estimated_qs(self):
        assert self._recording_finalized is True
        return self._record_future_q

    def __str__(self):
        return self.get_name()

    def __repr__(self):
        return self.__str__()


class LearningPlayer_v2(LearningPlayer):

    def get_game_result(self, game):
        return game.get_winbased_r()

    def _derive_feature_stack(self, game, player):
        # pad to achieve timestep length
        feature_stack = [np.concatenate([self.get_hand_vector(game, player),
                                         np.zeros(LearningPlayer.TIMESTEP_FEATURES - LearningPlayer.HAND_FEATURES)])]
        move_stack = [self.compute_move_vector(player, game.get_landlord_position(), move) for player, move in game.get_game_logs()]
        feature_stack.extend(move_stack)
        return np.vstack(feature_stack)

    def derive_features(self, game):
        return self._append_padding(self._derive_feature_stack(game, game.get_current_position())).astype(np.int8)

    def record_move(self, game, best_move, best_move_q, player: TurnPosition):
        history_matrix = self.derive_features(game)

        move_vector = self.compute_move_vector(player, game.get_landlord_position(), best_move)
        hand_vector = self.compute_remaining_hand_vector(game, move_vector, player)

        self.record_history_matrices.append(history_matrix)
        self.record_move_vectors.append(move_vector)
        self.record_hand_vectors.append(hand_vector)

        self._record_state_q.append(best_move_q)

    def compute_remaining_hand_vector(self, game, move_vector, player: TurnPosition):
        hand = game.get_hand(player)
        vector = np.zeros(len(Card) + 3)
        for i, card in enumerate(Card):
            vector[i] = hand.count(card) - move_vector[i]

        if game.is_betting_complete():
            vector[-3] = len(game.get_hand(game.get_landlord_position()))
            vector[-2] = len(game.get_hand(game.get_landlord_position().previous()))
            vector[-1] = len(game.get_hand(game.get_landlord_position().next()))
        else:
            vector[-3] = 17
            vector[-2] = 17
            vector[-1] = 17
        return vector

    def full_move_evaluation(self, game, legal_moves):
        history_features = self.derive_features(game)

        # all the moves we make from here will not affect the history, so assess it and copy
        history_vector = self.get_history_vector(history_features)
        history_matrix = np.tile(history_vector, (len(legal_moves), 1))

        # create features for each of the possible moves from this position
        move_options_matrix = np.vstack([self.compute_move_vector(game.get_current_position(),
                                game.get_landlord_position(), move) for move in legal_moves])

        # make remaining hand vectors for each of the possible moves from this position
        hand_matrix = np.vstack([self.compute_remaining_hand_vector(game,
                                move_vector, game.get_current_position()) for move_vector in move_options_matrix])

        predictions = self.get_position_predictions(history_matrix, move_options_matrix, hand_matrix)

        return predictions

class RandomPlayer(Player):
    def make_move(self, game, debug=False):
        legal_moves = game.get_legal_moves()
        return random.choice(legal_moves)

class NoBetPlayer(Player):
    def make_move(self, game, debug=False):
        return None


class InvalidMoveError(Exception):
    pass


class TypoError(Exception):
    pass


class HumanPlayer(Player):
    def __init__(self, name, reference_player=None, known_hand=False, ai_before=False):
        super().__init__(name)
        self.reference_player = reference_player
        self.first_turn = True
        self.known_hand = known_hand
        self.ai_before = ai_before

    def ask_ai(self, game, my_move, top_n=5):
        legal_moves = game.get_legal_moves()
        is_landlord = game.get_landlord_position() == game.get_current_position()
        predictions = self.reference_player.full_move_evaluation(game, legal_moves)

        sorted_moves = sorted(list(zip(legal_moves, predictions)), key=lambda x: -x[1] if is_landlord else x[1])

        print("\tEvaluation By", self.reference_player.get_name())
        if my_move == sorted_moves[0][0]:
            print("\tBest Move!")

        for i, move_tup in enumerate(sorted_moves):
            move, pred = move_tup
            move_marker = '<-----' if move == my_move else ''
            if i < top_n or move == my_move:
                print('\t', i, ": ", pred, move, move_marker)

        print('\n')

    @classmethod
    def cardstrings_to_cards(cls, card_strings):
        cards = []
        try:
            for card_string in card_strings:
                card_string = card_string.strip()
                try:
                    card_string = card_string.upper()
                except Exception:
                    pass
                cards.append(string_to_card(card_string))
        except ValueError:
            raise TypoError
        return cards

    @classmethod
    def parse_input_for_cardset(cls, input_string):
        if len(input_string) == 0:
            return None

        try:
            card_strings = input_string.split(' ')
        except Exception:
            raise TypoError

        return HumanPlayer.cardstrings_to_cards(card_strings)

    @classmethod
    def parse_input(cls, input_string, bet_phase=False):
        if bet_phase:
            try:
                bet_amount = int(input_string)
                return BetMove(bet_amount)
            except Exception:
                pass

        cards = HumanPlayer.parse_input_for_cardset(input_string)
        if cards is None:
            return None

        all_possible_moves = CardSet(Counter(cards)).get_all_moves()
        for move in all_possible_moves:
            if move.get_cards() == Counter(cards):
                return move
        raise InvalidMoveError

    def print_instructions(self):
        if self.first_turn:
            print("\n")
            print("Enter your cards separated by spaces. Example \"EIGHT EIGHT\".")
            print("Enter a number to make a bet (between 0 and 3) or Return to pass.")
            print("\n")
            self.first_turn = False
            self.print_instructions()
        else:
            print('\n')


    def decide_best_move(self, game):
        self.print_instructions()
        while True:

            print(self.get_name() + "'s turn!")
            if self.known_hand:
                print(game.get_hand(game.get_current_position()))

            # ai assistance
            if self.ai_before and self.known_hand:
                print("Asking AI", self.ai_before, self.known_hand)
                self.ask_ai(game, my_move=None)

            # get input
            inp = input(">").strip()
            try:
                human_move = HumanPlayer.parse_input(inp, bet_phase=not game.is_betting_complete())
                if human_move in game.get_legal_moves() or self.known_hand is False:
                    if not self.ai_before and self.known_hand:
                        self.ask_ai(game, human_move)
                    return human_move, 0
                else:
                    print('Illegal Move, please try again!')
            except TypoError:
                print("Invalid Input, check your spelling!")
            except InvalidMoveError:
                print("Not a valid Landlord Move!")



