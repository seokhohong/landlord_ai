from enum import Enum


class Card(Enum):
    THREE = ('THREE', '3', 0)
    FOUR = ('FOUR', '4', 1)
    FIVE = ('FIVE', '5', 2)
    SIX = ('SIX', '6', 3)
    SEVEN = ('SEVEN', '7', 4)
    EIGHT = ('EIGHT', '8', 5)
    NINE = ('NINE', '9', 6)
    TEN = ('TEN', '10', 7)
    JACK = ('JACK', 'J', 8)
    QUEEN = ('QUEEN', 'Q', 9)
    KING = ('KING', 'K', 10)
    ACE = ('ACE', 'A', 11)
    TWO = ('TWO', '2', 13)
    LITTLE_JOKER = ('LITTLE_JOKER', 'LJ', 14)
    BIG_JOKER = ('BIG_JOKER', 'BJ', 15)

    def __init__(self, name, shorthand, value):
        self.card_name = name
        self.shorthand = shorthand
        self.card_value = value

    def beats(self, other):
        return self.card_value > other.card_value

    def __gt__(self, other):
        return self.beats(other)

    def __lt__(self, other):
        return not self.beats(other)

    def get_name(self):
        return self.card_name

    def get_shorthand(self):
        return self.shorthand

    def next(self):
        return NEXT_MAP[self]

    def __str__(self):
        return self.card_name

    def __repr__(self):
        return self.__str__()


def string_to_card(str_to_card):
    for card in list(Card):
        if card.get_name() == str_to_card:
            return card
        if card.get_shorthand() == str_to_card:
            return card
    raise ValueError

NEXT_MAP = {
    Card.THREE: Card.FOUR,
    Card.FOUR: Card.FIVE,
    Card.FIVE: Card.SIX,
    Card.SIX: Card.SEVEN,
    Card.SEVEN: Card.EIGHT,
    Card.EIGHT: Card.NINE,
    Card.NINE: Card.TEN,
    Card.TEN: Card.JACK,
    Card.JACK: Card.QUEEN,
    Card.QUEEN: Card.KING,
    Card.KING: Card.ACE,
    Card.ACE: None,
    Card.TWO: None,
    Card.LITTLE_JOKER: None,
    Card.BIG_JOKER: None
}