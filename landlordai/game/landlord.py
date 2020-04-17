import random
from collections import Counter
from copy import copy

import numpy as np

from landlordai.game.card import Card
from landlordai.game.deck import LandlordDeck, CardSet
from landlordai.game.move import SpecificMove, BetMove, KittyReveal
from landlordai.game.player import TurnPosition


class LandlordGame:
    MAX_BET = 3
    NUM_PLAYERS = 3
    KITTY_SIZE = 3
    SWEEP_MULTIPLIER = 3
    DEAL_SIZE = 17
    # games shouldn't go this long anyway
    TURN_LIMIT = 99
    # kitty_callback should return a list of 3 cards; used if we want manual setting of cards
    def __init__(self, players, kitty_callback=None):
        self._players = players
        self._scores = [0] * 3
        self.string_logs = []
        self._move_logs = []
        assert(len(self._players) == LandlordGame.NUM_PLAYERS)
        self._setup()
        self._betting_complete = False
        self._kitty_callback = kitty_callback

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result._move_logs = [(copy(elem[0]), copy(elem[1])) for elem in self._move_logs]
        result.string_logs = copy(self.string_logs)
        result._scores = copy(self._scores)
        result._peasant_positions = copy(self._peasant_positions)
        result._hands = dict((k, copy(v)) for (k, v) in self._hands.items())
        return result

    def play_round(self, debug=False):
        self._setup()
        self._bet_rounds(debug=debug)
        if self._round_over:
            return None
        self.main_game(debug=debug)

    def _setup(self):
        self._landlord_position = None
        self._peasant_positions = []
        self._bet_amount = 0
        self._current_position = random.choice(list(TurnPosition))
        self._control_position = None
        self._round_over = False
        # list of winners
        self._winners = None
        deck = LandlordDeck()
        self.kitty = deck.draw(LandlordGame.KITTY_SIZE)
        self._hands = {TurnPosition.FIRST: deck.draw(LandlordGame.DEAL_SIZE),
                       TurnPosition.SECOND: deck.draw(LandlordGame.DEAL_SIZE),
                       TurnPosition.THIRD: deck.draw(LandlordGame.DEAL_SIZE)}


    def force_setup(self, landlord_position: TurnPosition, hands: dict, bet_amount: int):
        self._landlord_position = landlord_position
        self._current_position = self._landlord_position
        self._hands = hands
        self._bet_amount = bet_amount
        self._set_peasants()

    def force_current_position(self, current_position):
        self._current_position = current_position

    def force_kitty(self, kitty):
        self.kitty = kitty

    def force_hand(self, position: TurnPosition, hand):
        self._hands[position] = hand

    def _reveal_kitty(self):
        if self._kitty_callback is not None:
            self.kitty = self._kitty_callback()
            assert type(self.kitty) == list
            assert len(self.kitty) == 3
            assert type(self.kitty[0]) == Card
        # add the kitty to the landlord's hand
        self._hands[self._landlord_position] += self.kitty
        self._hands[self._landlord_position] = sorted(self._hands[self._landlord_position])
        self._move_logs.append((self._current_position, KittyReveal(self.kitty)))
        assert len(self.get_hand(self._landlord_position)) == LandlordGame.KITTY_SIZE + LandlordGame.DEAL_SIZE

    def _bet_rounds(self, debug=False):
        # limit number of steps to check for draw
        while not self._betting_complete:
            bet = self.get_current_player().make_move(self, debug=debug)
            self._make_bet_move(bet)


    '''
    def step_move(self, move):
        if type(move) == BetMove:
            self.make_bet(move)
            # decide if revealing kitty is an obligatory move
            if move.get_amount() == LandlordGame.MAX_BET or (self.move_logs[-1] is None and self.move_logs[-2] is None):
                self.reveal_kitty()
        if type(move) == SpecificMove:
            self.play_move(move)
    '''

    def _check_betting_complete(self, bet):
        if bet is not None and bet.get_amount() == LandlordGame.MAX_BET:
            self._current_position = self._landlord_position
            self._betting_complete = True

        if self.get_num_moves() >= LandlordGame.NUM_PLAYERS:
            # landlord position was decided earlier
            self._betting_complete = True

            # if nobody bet
            if self._bet_amount == 0:
                self._round_over = True
                return

            # everyone's had a chance to bet
            return

    def _make_bet_move(self, bet):
        if bet is not None and bet.get_amount() > self._bet_amount:
            self.string_logs.append(str(self._current_position) + " bet " + str(bet))
            self._move_logs.append((self._current_position, bet))
            self._bet_amount = bet.get_amount()
            self._landlord_position = self._current_position
        else:
            self.string_logs.append(str(self._current_position) + " passed")
            self._move_logs.append((self._current_position, None))

        self._current_position = self._current_position.next()

        self._check_betting_complete(bet)

        if self.is_betting_complete() and not self.is_round_over():
            self._reveal_kitty()
            self._set_peasants()
            self._current_position = self._landlord_position



    def get_num_moves(self):
        return len(self.get_move_logs())

    def _set_peasants(self):
        for position in list(TurnPosition):
            if position != self._landlord_position:
                self._peasant_positions.append(position)

    def is_betting_complete(self):
        return self._betting_complete

    def get_current_player(self):
        return self._players[self._current_position]

    def is_current_player_landlord(self):
        return self._current_position == self._landlord_position

    def get_current_position(self):
        return self._current_position

    def get_landlord_position(self):
        return self._landlord_position

    def get_hand(self, player: TurnPosition):
        return self._hands[player]

    def get_legal_moves(self):
        if self.is_betting_complete():
            hand = CardSet(Counter(self.get_hand(self.get_current_position())))
            all_moves = hand.get_all_moves()

            # you can play anything if you have control
            if self._control_position == self.get_current_position():
                return all_moves

            # otherwise you have to play moves that beat it, or pass
            return [move for move in all_moves if move.beats(self.get_last_played())] + [None]
        else:
            return [BetMove(x) for x in range(LandlordGame.MAX_BET + 1)]

    def get_game_logs(self):
        return self._move_logs

    def play_from_hand(self, move: SpecificMove, hand_known=True):
        hand = self.get_hand(self._current_position)
        for card, count in move.cards.items():
            for i in range(count):
                if hand_known:
                    hand.remove(card)
                else:
                    # if we don't know the hand, then just remove one card from it
                    hand = hand[1:]

    # main play_move, triages depending on move
    def play_move(self, move, hand_known=True):
        if self.is_betting_complete():
            self._make_card_move(move, hand_known)
        else:
            self._make_bet_move(move)

    def _make_card_move(self, move, hand_known=True):
        if move is not None:
            assert (move.beats(self.get_last_played()) or self._current_position == self._control_position)
            self.string_logs.append(str(self._current_position) + " played " + str(move))

            self.play_from_hand(move, hand_known=hand_known)
            if move.is_bomb():
                self._bet_amount = self._bet_amount * 2
            self._control_position = self._current_position
        else:
            self.string_logs.append(str(self._current_position) + " passed.")

        self._move_logs.append((self._current_position, move))
        self.compute_round_over()

        if not self.is_round_over():
            self._current_position = self._current_position.next()

    def compute_round_over(self):
        # game is over
        #print('cards', len(self.get_hand(self.current_position)))
        if len(self.get_hand(self._current_position)) == 0:
            #print('Over')
            self._round_over = True
            if self.peasants_have_no_plays():
                self._bet_amount *= LandlordGame.SWEEP_MULTIPLIER

            if self._current_position == self._landlord_position:
                self.string_logs.append(str(self._current_position) + " wins as Landlord!")
                self._winners = [self._landlord_position]
                # landlordai gains
                self._scores[self._current_position] += self._bet_amount * 2
                self._scores[self._peasant_positions[0]] -= self._bet_amount
                self._scores[self._peasant_positions[1]] -= self._bet_amount
            else:
                self.string_logs.append(str(self._peasant_positions) + " win as Peasants!")
                self._winners = self._peasant_positions
                # peasants gain
                self._scores[self._current_position] -= self._bet_amount * 2
                self._scores[self._peasant_positions[0]] += self._bet_amount
                self._scores[self._peasant_positions[1]] += self._bet_amount
            self.string_logs.append(str(self._scores))
            assert (sum(self._scores) == 0)
            return True
        return False

    def move_ends_game(self, move):
        player = self.get_current_position()
        if move is not None:
            if type(move) == SpecificMove:
                return self.get_current_position() == player and move.cards == Counter(self.get_hand(player))
            if type(move) == BetMove and move.get_amount() == 0 \
                    and self.get_num_moves() >= LandlordGame.NUM_PLAYERS - 1 and self.get_bet_amount() == 0:
                return True
        return False

    def player_has_won(self, position: TurnPosition):
        return np.argmax(self._scores) == position.index()

    def get_position_role_name(self, position: TurnPosition):
        if not self.is_betting_complete():
            return 'UNDECIDED'
        if position == self._landlord_position:
            return 'LANDLORD'
        return 'PEASANT'

    # used for the extra landlord bonus
    def peasants_have_no_plays(self):
        if self.is_betting_complete():
            if len(self._hands[self._peasant_positions[0]]) == LandlordGame.DEAL_SIZE and \
                len(self._hands[self._peasant_positions[1]]) == LandlordGame.DEAL_SIZE:
                return True
        return False

    def get_r(self):
        # the game never got played
        if self._winners is None:
            return 0

        # landlord win is positive
        if self._landlord_position in self._winners:
            return self._bet_amount * 2

        # peasant win is negative
        return - self._bet_amount

    # -1 to 1, -1 is peasant win
    def get_winbased_r(self):
        # the game never got played
        if self._winners is None:
            return 0

        # landlord win is positive
        if self._landlord_position in self._winners:
            return 1

        # peasant win is negative
        return -1

    def get_scores(self):
        return copy(self._scores)

    def main_game(self, debug=False):
        self._control_position = self._landlord_position
        self._current_position = self._landlord_position
        while True:
            move = self.get_current_player().make_move(self, debug=debug)
            self.play_move(move)
            #print(self.hands)
            if len(self.get_move_logs()) >= LandlordGame.TURN_LIMIT:
                break
            if self.is_round_over():
                break

    def get_bet_amount(self):
        return self._bet_amount

    def get_last_played(self):
        if len(self.get_move_logs()) == 0:
            return None
        for i in range(len(self.get_move_logs()) - 1, -1, -1):
            if self.get_move_logs()[i][1] is not None:
                return self.get_move_logs()[i][1]

        return None

    def get_control_position(self):
        return self._control_position

    def get_move_logs(self):
        return self._move_logs

    def is_round_over(self):
        return self._round_over

    def has_winners(self):
        return self._winners is not None

    def get_winners(self):
        return self._winners

    def get_winner_ais(self):
        return [self.get_ai(pos) for pos in self._winners]

    def get_loser_ais(self):
        return [self.get_ai(pos) for pos in list(TurnPosition) if pos not in self._winners]

    def get_ai(self, pos: TurnPosition):
        return self._players[pos]

    def get_ai_players(self):
        return self._players

