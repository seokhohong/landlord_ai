import random

from src.game.player import TurnPosition
from src.game.deck import LandlordDeck
from src.game.move import SpecificMove, BetMove, KittyReveal


class LandlordGame:
    MAX_BET = 3
    NUM_PLAYERS = 3
    KITTY_SIZE = 3
    DEAL_SIZE = 17
    def __init__(self, players, num_rounds=10):
        self.num_rounds = num_rounds
        self.players = players
        self.scores = [0] * 3
        self.string_logs = []
        self.move_logs = []
        assert(len(self.players) == LandlordGame.NUM_PLAYERS)
        self.setup()

    def play_rounds(self):
        for _ in range(self.num_rounds):
            self.play_round()

    def play_round(self):
        self.setup()
        self.bet()
        self.set_peasants()
        if self.round_over:
            return None
        self.main_game()

    def setup(self):
        self.last_played = None
        self.landlord_position = None
        self.peasant_positions = []
        self.bet_amount = 0
        self.starting_position = random.choice(list(TurnPosition))
        self.current_position = self.starting_position
        self.control_position = None
        self.round_over = False
        # list of winners
        self.winners = None
        self.kitty = []
        # list of lists
        self.hands = []
        deck = LandlordDeck()
        self.kitty = deck.draw(LandlordGame.KITTY_SIZE)
        self.hands = [deck.draw(LandlordGame.DEAL_SIZE), deck.draw(LandlordGame.DEAL_SIZE), deck.draw(LandlordGame.DEAL_SIZE)]

    def force_setup(self, landlord_position, hands):
        self.landlord_position = landlord_position
        self.hands = hands
        self.set_peasants()

    def bet(self):
        # limit number of steps to check for draw
        for i in range(6):
            bet = self.get_current_player().make_bet(self)
            if bet is not None and bet > self.bet_amount:
                self.string_logs.append(str(self.current_position) + " bet " + str(bet))
                self.move_logs.append((self.current_position, BetMove(bet)))
                self.bet_amount = bet
                self.landlord_position = self.current_position
            else:
                self.string_logs.append(str(self.current_position) + " passed")
                self.move_logs.append((self.current_position, None))
            if bet == LandlordGame.MAX_BET:
                break
            self.current_position = self.current_position.next()

        # if nobody bet
        if self.bet_amount == 0:
            self.round_over = True
            return

        # add the kitty to the landlord's hand
        self.hands[self.landlord_position] += self.kitty
        self.move_logs.append((self.current_position, KittyReveal(self.kitty)))
        assert len(self.get_hand(self.landlord_position)) == LandlordGame.KITTY_SIZE + LandlordGame.DEAL_SIZE

    def set_peasants(self):
        for position in list(TurnPosition):
            if position != self.landlord_position:
                self.peasant_positions.append(position)

    def get_current_player(self):
        return self.players[self.current_position]

    def current_player_is_landlord(self):
        return self.current_position == self.landlord_position

    def get_current_position(self):
        return self.current_position

    def get_landlord_position(self):
        return self.landlord_position

    def get_hand(self, player: TurnPosition):
        return self.hands[player]

    def play_from_hand(self, player: TurnPosition, move: SpecificMove):
        hand = self.get_hand(player)
        for card, count in move.cards.items():
            for i in range(count):
                hand.remove(card)

    def play_move(self, move):
        self.move_logs.append(move)
        if move is not None:
            assert (move.beats(self.last_played) or self.current_position == self.control_position)
            self.last_played = move
            self.string_logs.append(str(self.current_position) + " played " + str(move))

            self.play_from_hand(self.current_position, move)
            self.control_position = self.current_position
        else:
            self.string_logs.append(str(self.current_position) + " passed.")

    def is_game_over(self):
        # game is over
        if len(self.get_hand(self.current_position)) == 0:
            self.round_over = True
            if self.current_position == self.landlord_position:
                self.string_logs.append(str(self.current_position) + " wins as Landlord!")
                self.winners = [self.landlord_position]
                # landlord gains
                self.scores[self.current_position] += self.bet_amount * 2
                self.scores[self.peasant_positions[0]] -= self.bet_amount
                self.scores[self.peasant_positions[1]] -= self.bet_amount
            else:
                self.string_logs.append(str(self.peasant_positions) + " win as Peasants!")
                self.winners = self.peasant_positions
                # peasants gain
                self.scores[self.current_position] -= self.bet_amount * 2
                self.scores[self.peasant_positions[0]] += self.bet_amount
                self.scores[self.peasant_positions[1]] += self.bet_amount
            self.string_logs.append(str(self.scores))
            assert (sum(self.scores) == 0)
            return True
        return False

    def main_game(self):
        self.control_position = self.landlord_position
        self.current_position = self.landlord_position
        while True:
            move = self.get_current_player().make_move(self, self.current_position)
            self.play_move(move)
            if self.is_game_over():
                break
            # next player moves
            self.current_position = self.current_position.next()


    def get_bet_amount(self):
        return self.bet_amount

    def get_last_played(self):
        return self.last_played

    def get_control_position(self):
        return self.control_position

    def get_move_logs(self):
        return self.move_logs

