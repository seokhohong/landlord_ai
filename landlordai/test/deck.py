import unittest
from collections import Counter

from landlordai.game.card import Card
from landlordai.game.deck import CardSet
from landlordai.game.move import RankedMoveType, MoveType, BetMove, KittyReveal


class TestLandlordMethods(unittest.TestCase):

    def test_card(self):
        self.assertTrue(Card.JACK.beats(Card.TEN))
        self.assertTrue(Card.BIG_JOKER.beats(Card.LITTLE_JOKER))
        self.assertFalse(Card.TWO.beats(Card.TWO))

    def test_move(self):
        self.assertTrue(RankedMoveType(MoveType.CHAIN_PAIR_4, Card.KING).beats(RankedMoveType(MoveType.CHAIN_PAIR_4, Card.QUEEN)))
        self.assertFalse(RankedMoveType(MoveType.BOMB, Card.KING).beats(RankedMoveType(MoveType.BOMB, Card.BIG_JOKER)))
        self.assertTrue(RankedMoveType(MoveType.BOMB, Card.THREE).beats(RankedMoveType(MoveType.TRIPLE_PAIR_KICKER, Card.FIVE)))
        self.assertTrue(RankedMoveType(MoveType.PAIR, Card.ACE).beats(RankedMoveType(MoveType.PAIR, Card.JACK)))
        self.assertFalse(RankedMoveType(MoveType.PAIR, Card.ACE).beats(RankedMoveType(MoveType.PAIR, Card.ACE)))
        self.assertFalse(RankedMoveType(MoveType.TRIPLE_PAIR_KICKER, Card.FOUR).beats(RankedMoveType(MoveType.TRIPLE_SINGLE_KICKER, Card.TEN)))
        self.assertTrue(RankedMoveType(MoveType.SINGLE, Card.FOUR).beats(RankedMoveType(MoveType.SINGLE, Card.THREE)))

    def test_simple_hand(self):
        hand = CardSet(Counter({Card.JACK: 3, Card.QUEEN: 1}))
        self.assertTrue(len(hand._get_single_moves()) == 2)
        self.assertTrue(len(hand._get_pair_moves()) == 1)
        self.assertTrue(len(hand._get_triple_moves()) == 2)

    def test_simple_hand_2(self):
        hand = CardSet(Counter({Card.JACK: 3, Card.QUEEN: 2, Card.KING: 3}))
        self.assertTrue(len(hand._get_single_moves()) == 3)
        self.assertTrue(len(hand._get_pair_moves()) == 3)
        self.assertTrue(len(hand._get_triple_moves()) == 10)

    def test_airplane(self):
        hand = CardSet(Counter({Card.JACK: 3, Card.QUEEN: 2, Card.KING: 3}))
        self.assertTrue(len(hand._get_airplanes()) == 0)
        hand = CardSet(Counter({Card.JACK: 3, Card.QUEEN: 3, Card.THREE: 2, Card.TWO: 2}))
        self.assertTrue(len(hand._get_airplanes()) == 2)

    def test_straight(self):
        hand = CardSet(Counter({Card.THREE: 2, Card.FOUR: 1, Card.FIVE: 1, Card.SIX: 1, Card.SEVEN: 1,
                                Card.EIGHT: 1, Card.NINE: 2}))
        self.assertTrue(len(hand._get_straights()) == 6)
        hand = CardSet(Counter({Card.THREE: 2, Card.FOUR: 2, Card.FIVE: 3, Card.SIX: 2, Card.SEVEN: 2,
                                Card.EIGHT: 2, Card.NINE: 2}))
        self.assertTrue(len(hand._get_straights()) == 21)

        hand = CardSet(Counter({Card.SEVEN: 1, Card.NINE: 2, Card.TEN: 1, Card.JACK: 2, Card.QUEEN: 2,
                                Card.KING: 2}))
        self.assertTrue(len(hand._get_straights()) == 2)
        self.assertTrue(len(hand._get_straights()[0].cards) == 5)

    def test_quads(self):
        hand = CardSet(Counter({Card.JACK: 4, Card.QUEEN: 2, Card.KING: 3}))
        self.assertTrue(len(hand._get_quad_moves()) == 3)

    def test_complex(self):
        hand = CardSet(Counter({Card.THREE: 3, Card.FOUR: 1, Card.FIVE: 2, Card.SIX: 1, Card.SEVEN: 1,
                                Card.LITTLE_JOKER: 1, Card.BIG_JOKER: 1}))
        self.assertTrue(len(hand.get_all_moves()) == 19)

    def test_chains(self):
        hand = CardSet(Counter({Card.THREE: 2, Card.FOUR: 3, Card.FIVE: 2}))
        self.assertTrue(len(hand._get_straights_from(Card.THREE, num_cards=2)) == 1)

    def test_chains2(self):
        hand = CardSet(Counter({Card.THREE: 2, Card.FOUR: 3, Card.FIVE: 2, Card.SIX: 2}))
        self.assertTrue(len(hand._get_straights_from(Card.THREE, num_cards=2)) == 2)

    def test_joker_kickers(self):
        hand = CardSet(Counter({Card.THREE: 3, Card.FOUR: 3,
                                Card.LITTLE_JOKER: 1, Card.BIG_JOKER: 1}))
        self.assertEqual(len(hand.get_all_moves()), 17)

    def test_equality(self):
        self.assertEqual(BetMove(3), BetMove(3))
        self.assertNotEqual(BetMove(2), BetMove(0))
        self.assertEqual(KittyReveal(cards=[Card.FIVE, Card.SIX, Card.SEVEN]),
                         KittyReveal(cards=[Card.FIVE, Card.SIX, Card.SEVEN]))
        self.assertNotEqual(KittyReveal(cards=[Card.FIVE, Card.SIX, Card.SEVEN]),
                         KittyReveal(cards=[Card.FIVE, Card.KING, Card.SEVEN]))
        self.assertNotEqual(BetMove(3), KittyReveal(cards=[Card.FIVE, Card.SIX, Card.SEVEN]))

if __name__ == '__main__':
    unittest.main()

