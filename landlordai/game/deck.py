import random
from collections import Counter

from landlordai.game.card import Card
from landlordai.game.move import SpecificMove, RankedMoveType, MoveType


class CardSet:
    def __init__(self, cards: Counter):
        self._cards = cards

    def remove(self, cards: Counter):
        copy_self = self._cards.copy()
        copy_self.subtract(cards)
        return CardSet(copy_self)

    def get_all_moves(self):
        moves = []
        moves.extend(self._get_single_moves())
        moves.extend(self._get_pair_moves())
        moves.extend(self._get_triple_moves())
        moves.extend(self._get_quad_moves())
        moves.extend(self._get_straights())
        moves.extend(self._get_airplanes())
        moves.extend(self._get_special_moves())
        return moves

    def _get_single_moves(self):
        moves = []
        for card in self._cards:
            moves.append(SpecificMove(RankedMoveType(MoveType.SINGLE, card), Counter({card: 1})))
        return moves

    def _get_cards_of_count(self, count):
        cards = []
        for card in self._cards:
            if self._cards[card] >= count:
                cards.append(card)
        return cards

    def _get_pair_moves(self):
        moves = []
        for card in self._cards:
            if self._cards[card] >= 2:
                moves.append(SpecificMove(RankedMoveType(MoveType.PAIR, card), Counter({card: 2})))
        return moves

    def _get_triple_moves(self):
        moves = []
        for core_card in self._cards:
            if self._cards[core_card] >= 3:
                hand_without_triple = self.remove(Counter({core_card: 3}))
                # no kickers
                moves.append(SpecificMove(RankedMoveType(MoveType.TRIPLE, core_card), Counter({core_card: 3})))
                # look for single kickers
                for move in hand_without_triple._get_single_moves():
                    # kicker isn't the same as the core
                    kicker = move.rank()
                    if kicker != core_card:
                        moves.append(SpecificMove(RankedMoveType(MoveType.TRIPLE_SINGLE_KICKER, core_card),
                                                  Counter({core_card: 3, kicker: 1})))
                for move in hand_without_triple._get_pair_moves():
                    kicker = move.rank()
                    if kicker != core_card:
                        moves.append(SpecificMove(RankedMoveType(MoveType.TRIPLE_PAIR_KICKER, core_card),
                                                  Counter({core_card: 3, kicker: 2})))
        return moves

    # kicker_n=2 means pair kickers
    def _get_two_kickers(self, kicker_n=1):

        kickers = set()
        for kicker_1 in self._get_cards_of_count(kicker_n):
            use_a_kicker = self.remove(Counter({kicker_1: kicker_n}))
            for kicker_2 in use_a_kicker._get_cards_of_count(kicker_n):
                if kicker_1 != kicker_2 and sorted([kicker_1, kicker_2]) != [Card.LITTLE_JOKER, Card.BIG_JOKER]:
                    kickers.add(tuple(sorted([kicker_1, kicker_2])))
        return kickers

    def _get_quad_moves(self):
        moves = []
        for core_card in self._cards:
            if self._cards[core_card] == 4:
                hand_without_core = self.remove(Counter({core_card: 4}))
                # just as bomb
                moves.append(SpecificMove(RankedMoveType(MoveType.BOMB, core_card), Counter({core_card: 4})))
                # two single kickers
                for kicker1, kicker2 in hand_without_core._get_two_kickers():
                    moves.append(SpecificMove(RankedMoveType(MoveType.QUAD_SINGLE_KICKERS, core_card), Counter({core_card: 4, kicker1: 1, kicker2: 1})))
                for kicker1, kicker2 in hand_without_core._get_two_kickers(kicker_n=2):
                    moves.append(SpecificMove(RankedMoveType(MoveType.QUAD_SINGLE_KICKERS, core_card),
                                              Counter({core_card: 4, kicker1: 2, kicker2: 2})))
        return moves

    # use num_cards=2 for pairs, num_cards=3 for triples, etc.
    def _get_straights_from(self, card, num_cards=1):
        consecutives_required = 5 if num_cards == 1 else 3
        moves = []
        consecutive = 1
        cards_included = Counter({card: num_cards})
        while card.next() is not None:
            if self._cards[card.next()] >= num_cards:
                consecutive += 1
                cards_included[card.next()] = num_cards
                card = card.next()
            else:
                break
            if consecutive >= consecutives_required:
                assert(len(cards_included) == consecutive)
                moves.append(SpecificMove(RankedMoveType(MoveType.get_straight_of_length(consecutive, num_cards), card), cards_included.copy()))
        return moves

    def _get_straights(self):
        moves = []
        for i, card in enumerate(LandlordDeck.NORMAL_CARD_TYPES):
            # search for chains up to triples
            for j in range(min(self._cards[card], 3)):
                moves.extend(self._get_straights_from(card, j + 1))
        return moves

    def _get_airplanes(self):
        # checking only 2-consecutive
        moves = []
        for i, card in enumerate(LandlordDeck.NORMAL_CARD_TYPES):
            if self._cards[card] >= 3 and card.next() is not None and self._cards[card.next()] >= 3:
                airplane_cards = Counter({card: 3, card.next(): 3})
                hand_without_core = self.remove(airplane_cards)
                for kicker1, kicker2 in hand_without_core._get_two_kickers():
                    specific_cards = airplane_cards + Counter({kicker1: 1, kicker2: 1})
                    moves.append(
                        SpecificMove(RankedMoveType(MoveType.AIRPLANE_SINGLE_KICKER, card.next()), specific_cards))

                for kicker1, kicker2 in hand_without_core._get_two_kickers(kicker_n=2):
                    specific_cards = airplane_cards + Counter({kicker1: 2, kicker2: 2})
                    moves.append(
                        SpecificMove(RankedMoveType(MoveType.AIRPLANE_SINGLE_KICKER, card.next()), specific_cards))
        return moves

    def _get_special_moves(self):
        # rocket
        moves = []
        if self._cards[Card.LITTLE_JOKER] == 1 and self._cards[Card.BIG_JOKER] == 1:
            moves.append(SpecificMove(RankedMoveType(MoveType.BOMB, Card.BIG_JOKER), Counter({Card.LITTLE_JOKER: 1, Card.BIG_JOKER: 1})))
        return moves


class LandlordDeck:
    NORMAL_CARD_TYPES = [
        Card.THREE, Card.FOUR, Card.FIVE, Card.SIX, Card.SEVEN, Card.EIGHT, Card.NINE,
        Card.TEN, Card.JACK, Card.QUEEN, Card.KING, Card.ACE, Card.TWO
    ]
    EXTRA_CARD_TYPES = [
        Card.LITTLE_JOKER, Card.BIG_JOKER
    ]
    NUM_CARDS = 54
    def __init__(self):
        self.cards = []
        for card in LandlordDeck.NORMAL_CARD_TYPES:
            self.cards.extend([card] * 4)
        for card in LandlordDeck.EXTRA_CARD_TYPES:
            self.cards.append(card)
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, num_cards):
        drawn = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return sorted(drawn)

