from collections import Counter
from enum import Enum, auto
from copy import copy

from landlordai.game.card import Card


class BetMove:
    def __init__(self, amount: int):
        assert type(amount) == int
        self.amount = amount

    def get_amount(self):
        return self.amount

    def __str__(self):
        return "BetMove (" + str(self.amount) + ")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if other is None or type(other) != BetMove:
            return False
        return self.amount == other.amount

class KittyReveal:
    def __init__(self, cards: list):
        self.cards = cards

    def __str__(self):
        return "Kitty: " + ','.join([str(card) for card in self.cards])

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if other is None or type(other) != KittyReveal:
            return False

        return self.cards == other.cards

class MoveType(Enum):
    SINGLE = auto(),
    PAIR = auto(),
    STRAIGHT_5 = auto(),
    STRAIGHT_6 = auto(),
    STRAIGHT_7 = auto(),
    STRAIGHT_8 = auto(),
    STRAIGHT_9 = auto(),
    STRAIGHT_10 = auto(),
    STRAIGHT_11 = auto(),
    STRAIGHT_12 = auto(),
    TRIPLE = auto(),
    TRIPLE_SINGLE_KICKER = auto(),
    TRIPLE_PAIR_KICKER = auto(),
    AIRPLANE_SINGLE_KICKER = auto(),
    AIRPLANE_PAIR_KICKER = auto(),
    CHAIN_PAIR_3 = auto(),
    CHAIN_PAIR_4 = auto(),
    CHAIN_PAIR_5 = auto(),
    CHAIN_PAIR_6 = auto(),
    CHAIN_PAIR_7 = auto(),
    CHAIN_PAIR_8 = auto(),
    CHAIN_PAIR_9 = auto(),
    CHAIN_PAIR_10 = auto(),
    CHAIN_PAIR_11 = auto(),
    CHAIN_PAIR_12 = auto(),
    CHAIN_TRIPLE_3 = auto(),
    CHAIN_TRIPLE_4 = auto(),
    CHAIN_TRIPLE_5 = auto(),
    CHAIN_TRIPLE_6 = auto(),
    QUAD_SINGLE_KICKERS = auto(),
    QUAD_PAIR_KICKERS = auto(),
    BOMB = auto()

    @classmethod
    def get_straight_of_length(cls, l, num_cards):
        if num_cards == 1:
            if l == 5: return MoveType.STRAIGHT_5
            if l == 6: return MoveType.STRAIGHT_6
            if l == 7: return MoveType.STRAIGHT_7
            if l == 8: return MoveType.STRAIGHT_8
            if l == 9: return MoveType.STRAIGHT_9
            if l == 10: return MoveType.STRAIGHT_10
            if l == 11: return MoveType.STRAIGHT_11
            if l == 12: return MoveType.STRAIGHT_12
        elif num_cards == 2:
            if l == 3: return MoveType.CHAIN_PAIR_3
            if l == 4: return MoveType.CHAIN_PAIR_4
            if l == 5: return MoveType.CHAIN_PAIR_5
            if l == 6: return MoveType.CHAIN_PAIR_6
            if l == 7: return MoveType.CHAIN_PAIR_7
            if l == 8: return MoveType.CHAIN_PAIR_8
            if l == 9: return MoveType.CHAIN_PAIR_9
            if l == 10: return MoveType.CHAIN_PAIR_10
            if l == 11: return MoveType.CHAIN_PAIR_11
            if l == 12: return MoveType.CHAIN_PAIR_12
        elif num_cards == 3:
            if l == 3: return MoveType.CHAIN_TRIPLE_3
            if l == 4: return MoveType.CHAIN_TRIPLE_4
            if l == 5: return MoveType.CHAIN_TRIPLE_5
            if l == 6: return MoveType.CHAIN_TRIPLE_6


# rank_card is the card that gives the move it's strength, i.e. the highest card of the straight or the non-kicker card
class RankedMoveType:
    def __init__(self, move_type: MoveType, rank_card: Card):
        self.move_type = move_type
        self.rank_card = rank_card

    def beats(self, other):
        if self.move_type != other.move_type:
            if self.move_type == MoveType.BOMB:
                return True
        else:
            return self.rank_card.beats(other.rank_card)

    def __str__(self):
        return self.move_type.name + '/' + self.rank_card.name


class SpecificMove:
    def __init__(self, ranked_move_type: RankedMoveType, cards: Counter):
        self.ranked_move_type = ranked_move_type
        self.cards = cards
        # just an assert
        if ranked_move_type.move_type == MoveType.STRAIGHT_5 or ranked_move_type.move_type == MoveType.STRAIGHT_6:
            assert ranked_move_type.rank_card == max(cards.keys())

    def rank(self):
        return self.ranked_move_type.rank_card

    def beats(self, other):
        if other is None:
            return True
        if type(other) == KittyReveal:
            return True
        return self.ranked_move_type.beats(other.ranked_move_type)

    def is_bomb(self):
        return self.ranked_move_type.move_type == MoveType.BOMB

    def get_cards(self):
        return copy(self.cards)

    def __str__(self):
        return str(self.ranked_move_type) + '(' + str(self.cards) + ')'

    def __eq__(self, other):
        if other is None:
            return False
        return self.cards == other.cards

