import random
from collections import Counter
from copy import copy

import numpy as np

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
    def __init__(self, players):
        self.players = players
        self.scores = [0] * 3
        self.string_logs = []
        self.move_logs = []
        assert(len(self.players) == LandlordGame.NUM_PLAYERS)
        self.setup()
        self.betting_complete = False

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.move_logs = [(copy(elem[0]), copy(elem[1])) for elem in self.move_logs]
        result.string_logs = copy(self.string_logs)
        result.scores = copy(self.scores)
        result.peasant_positions = copy(self.peasant_positions)
        result.hands = dict((k, copy(v)) for (k, v) in self.hands.items())
        return result

    def play_round(self, debug=False):
        self.setup()
        self.bet_rounds(debug=debug)
        if self.round_over:
            return None
        self.main_game(debug=debug)

    def setup(self):
        self.landlord_position = None
        self.peasant_positions = []
        self.bet_amount = 0
        self.starting_position = random.choice(list(TurnPosition))
        self._current_position = self.starting_position
        self.control_position = None
        self.round_over = False
        # list of winners
        self.winners = None
        deck = LandlordDeck()
        self.kitty = deck.draw(LandlordGame.KITTY_SIZE)
        self.hands = {TurnPosition.FIRST: deck.draw(LandlordGame.DEAL_SIZE),
                      TurnPosition.SECOND: deck.draw(LandlordGame.DEAL_SIZE),
                      TurnPosition.THIRD: deck.draw(LandlordGame.DEAL_SIZE)}


    def force_setup(self, landlord_position: TurnPosition, hands: dict, bet_amount: int):
        self.landlord_position = landlord_position
        self._current_position = self.landlord_position
        self.hands = hands
        self.bet_amount = bet_amount
        self.set_peasants()

    def force_current_position(self, current_position):
        self._current_position = current_position

    def force_kitty(self, kitty):
        self.kitty = kitty

    def reveal_kitty(self):
        # add the kitty to the landlord's hand
        self.hands[self.landlord_position] += self.kitty
        self.hands[self.landlord_position] = sorted(self.hands[self.landlord_position])
        self.move_logs.append((self._current_position, KittyReveal(self.kitty)))
        assert len(self.get_hand(self.landlord_position)) == LandlordGame.KITTY_SIZE + LandlordGame.DEAL_SIZE

    def bet_rounds(self, debug=False):
        # limit number of steps to check for draw
        while not self.betting_complete:
            bet = self.get_current_player().make_move(self, debug=debug)
            self.make_bet_move(bet)


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

    def check_betting_complete(self, bet):
        if bet is not None and bet.get_amount() == LandlordGame.MAX_BET:
            self._current_position = self.landlord_position
            self.betting_complete = True

        if self.get_num_moves() >= LandlordGame.NUM_PLAYERS:
            # landlord position was decided earlier
            self.betting_complete = True

            # if nobody bet
            if self.bet_amount == 0:
                self.round_over = True
                return

            # everyone's had a chance to bet
            return

    def make_bet_move(self, bet):
        if bet is not None and bet.get_amount() > self.bet_amount:
            self.string_logs.append(str(self._current_position) + " bet " + str(bet))
            self.move_logs.append((self._current_position, bet))
            self.bet_amount = bet.get_amount()
            self.landlord_position = self._current_position
        else:
            self.string_logs.append(str(self._current_position) + " passed")
            self.move_logs.append((self._current_position, None))

        self._current_position = self._current_position.next()

        self.check_betting_complete(bet)

        if self.is_betting_complete() and not self.is_round_over():
            self.reveal_kitty()
            self.set_peasants()
            self._current_position = self.landlord_position



    def get_num_moves(self):
        return len(self.get_move_logs())

    def set_peasants(self):
        for position in list(TurnPosition):
            if position != self.landlord_position:
                self.peasant_positions.append(position)

    def is_betting_complete(self):
        return self.betting_complete

    def get_current_player(self):
        return self.players[self._current_position]

    def current_player_is_landlord(self):
        return self._current_position == self.landlord_position

    def get_current_position(self):
        return self._current_position

    def get_landlord_position(self):
        return self.landlord_position

    def get_hand(self, player: TurnPosition):
        return self.hands[player]

    def get_legal_moves(self):
        if self.is_betting_complete():
            hand = CardSet(Counter(self.get_hand(self.get_current_position())))
            all_moves = hand.get_all_moves()

            # you can play anything if you have control
            if self.control_position == self.get_current_position():
                return all_moves

            # otherwise you have to play moves that beat it, or pass
            return [move for move in all_moves if move.beats(self.get_last_played())] + [None]
        else:
            return [BetMove(0), BetMove(1), BetMove(2), BetMove(3)]

    def get_game_logs(self):
        return self.move_logs

    def play_from_hand(self, move: SpecificMove):
        hand = self.get_hand(self._current_position)
        for card, count in move.cards.items():
            for i in range(count):
                hand.remove(card)

    # main play_move, triages depending on move
    def play_move(self, move):
        if self.is_betting_complete():
            self.make_card_move(move)
        else:
            self.make_bet_move(move)

    def make_card_move(self, move):
        if move is not None:
            assert (move.beats(self.get_last_played()) or self._current_position == self.control_position)
            self.string_logs.append(str(self._current_position) + " played " + str(move))

            self.play_from_hand(move)
            if move.is_bomb():
                self.bet_amount = self.bet_amount * 2
            self.control_position = self._current_position
        else:
            self.string_logs.append(str(self._current_position) + " passed.")

        self.move_logs.append((self._current_position, move))
        self.compute_round_over()

        if not self.is_round_over():
            self._current_position = self._current_position.next()

    def compute_round_over(self):
        # game is over
        #print('cards', len(self.get_hand(self.current_position)))
        if len(self.get_hand(self._current_position)) == 0:
            #print('Over')
            self.round_over = True
            if self.peasants_have_no_plays():
                self.bet_amount *= LandlordGame.SWEEP_MULTIPLIER

            if self._current_position == self.landlord_position:
                self.string_logs.append(str(self._current_position) + " wins as Landlord!")
                self.winners = [self.landlord_position]
                # landlordai gains
                self.scores[self._current_position] += self.bet_amount * 2
                self.scores[self.peasant_positions[0]] -= self.bet_amount
                self.scores[self.peasant_positions[1]] -= self.bet_amount
            else:
                self.string_logs.append(str(self.peasant_positions) + " win as Peasants!")
                self.winners = self.peasant_positions
                # peasants gain
                self.scores[self._current_position] -= self.bet_amount * 2
                self.scores[self.peasant_positions[0]] += self.bet_amount
                self.scores[self.peasant_positions[1]] += self.bet_amount
            self.string_logs.append(str(self.scores))
            assert (sum(self.scores) == 0)
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
        return np.argmax(self.scores) == position.index()

    def get_position_role_name(self, position: TurnPosition):
        if not self.is_betting_complete():
            return 'UNDECIDED'
        if position == self.landlord_position:
            return 'LANDLORD'
        return 'PEASANT'

    # used for the extra landlord bonus
    def peasants_have_no_plays(self):
        if self.is_betting_complete():
            if len(self.hands[self.peasant_positions[0]]) == LandlordGame.DEAL_SIZE and \
                len(self.hands[self.peasant_positions[1]]) == LandlordGame.DEAL_SIZE:
                return True
        return False

    def get_r(self):
        # the game never got played
        if self.winners is None:
            return 0

        # landlord win is positive
        if self.landlord_position in self.winners:
            return self.bet_amount * 2

        # peasant win is negative
        return - self.bet_amount

    def get_scores(self):
        return copy(self.scores)

    def main_game(self, debug=False):
        self.control_position = self.landlord_position
        self._current_position = self.landlord_position
        while True:
            move = self.get_current_player().make_move(self, debug=debug)
            self.play_move(move)
            #print(self.hands)
            if len(self.get_move_logs()) >= LandlordGame.TURN_LIMIT:
                break
            if self.is_round_over():
                break

    def get_bet_amount(self):
        return self.bet_amount

    def get_last_played(self):
        if len(self.get_move_logs()) == 0:
            return None
        for i in range(len(self.get_move_logs()) - 1, -1, -1):
            if self.get_move_logs()[i][1] is not None:
                return self.get_move_logs()[i][1]

        return None

    def get_control_position(self):
        return self.control_position

    def get_move_logs(self):
        return self.move_logs

    def is_round_over(self):
        return self.round_over

    def has_winners(self):
        return self.winners is not None

    def get_winners(self):
        return self.winners

    def get_winner_ais(self):
        return [self.get_ai(pos) for pos in self.winners]

    def get_loser_ais(self):
        return [self.get_ai(pos) for pos in list(TurnPosition) if pos not in self.winners]

    def get_ai(self, pos: TurnPosition):
        return self.players[pos]

    def get_ai_players(self):
        return self.players

