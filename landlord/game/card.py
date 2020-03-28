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
    LITTLE_JOKER = ('LITTLE_JOKER', 'jk', 14)
    BIG_JOKER = ('BIG_JOKER', 'JK', 15)

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

    def next(self):
        if self == Card.THREE: return Card.FOUR
        if self == Card.FOUR: return Card.FIVE
        if self == Card.FIVE: return Card.SIX
        if self == Card.SIX: return Card.SEVEN
        if self == Card.SEVEN: return Card.EIGHT
        if self == Card.EIGHT: return Card.NINE
        if self == Card.NINE: return Card.TEN
        if self == Card.TEN: return Card.JACK
        if self == Card.JACK: return Card.QUEEN
        if self == Card.QUEEN: return Card.KING
        if self == Card.KING: return Card.ACE
        return None

    def __str__(self):
        return self.card_name

    def __repr__(self):
        return self.__str__()