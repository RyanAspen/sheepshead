import numpy as np
from numpy.typing import NDArray
from sheepshead.suit import Suit
import sheepshead.card_constants as card_constants

class Hand:

    def __init__(self):
        self.cards = np.zeros(32, dtype=np.int8)
        self.is_picker = False
        self.partner_revealed = False
        self.number_of_cards = 0
        self.suit_counts = {
            Suit.CLUB : 0,
            Suit.SPADE : 0,
            Suit.HEART : 0,
            Suit.TRUMP : 0
        }

    def __repr__(self):
        if self.number_of_cards == 0:
            return "EMPTY HAND"
        s = "HAND: "
        for i in range(32):
            if self.cards[i]:
                s = s + card_constants.card_ids[i] + ", "
        return s[:-2]

    def has_card(self, card_int : int) -> bool:
        return self.cards[card_int]

    def get_hand_mask(self) -> NDArray[np.int8]:
        return self.cards

    def get_cards_list(self) -> list[int]:
        return np.nonzero(self.cards)[0].tolist()

    def add_card(self, card_int : int):
        self.suit_counts[card_constants.card_suits[card_int]] += 1
        self.cards[card_int] = 1
        self.number_of_cards += 1

    def add_cards(self, card_ints : list[int]):
        for card_int in card_ints:
            self.add_card(card_int)

    def set_as_picker(self):
        self.is_picker = True

    def set_partner_revealed(self):
        self.partner_revealed = True

    def get_legal_called_aces(self) -> list[Suit]:
        legal_called_aces = []
        for suit in [Suit.CLUB, Suit.SPADE, Suit.HEART]:
            if self.suit_counts[suit] == 0:
                continue
            if self.cards[card_constants.aces[suit]] == 0:
                legal_called_aces.append(suit)
        return legal_called_aces
    
    def can_bury_card(self, card_int : int, called_ace : Suit | None) -> bool:
        legal_cards = self.get_legal_cards_to_bury(called_ace)
        return legal_cards[card_int] > 0

    def get_legal_cards_to_bury(self, called_ace : Suit | None) -> NDArray[np.bool]:
        if called_ace is None:
            return self.cards
        elif self.suit_counts[called_ace] > 1:
            return self.cards
        else:
            return (self.cards & np.logical_not(card_constants.card_suit_masks[called_ace])).astype(np.int8)

    # NOTE: This doesn't check for legality, returns score buried 
    def bury(self, card_int : int) -> int:
        self.cards[card_int] = 0
        self.suit_counts[card_constants.card_suits[card_int]] -= 1
        self.number_of_cards -= 1
        return card_constants.card_points[card_int]

    def can_play_card(self, card_int : int, led_suit : Suit | None, called_ace : Suit | None) -> bool:
        legal_cards = self.get_legal_cards_to_play(led_suit, called_ace)
        return legal_cards[card_int] > 0

    def get_legal_cards_to_play(self, led_suit : Suit | None, called_ace : Suit | None) -> NDArray[np.int8]:
        if self.number_of_cards == 1:
            # We have no choice
            return self.cards
        elif led_suit is None:
            # Can play anything
            return self.cards
        elif called_ace is None:
            # Must follow suit if possible, no other limits
            if self.suit_counts[led_suit] > 0:
                return (self.cards & card_constants.card_suit_masks[led_suit]).astype(np.int8)
            else:
                return self.cards
        else:
            called_ace_int = card_constants.aces[called_ace]
            has_called_ace = self.cards[called_ace_int]
            if has_called_ace:
                # If the called ace suit is led and we have the ace, we must play it
                if led_suit == called_ace:
                    return card_constants.ace_masks[called_ace]
                # If the called ace suit was not led and we have the ace, we cannot play it unless we have no other option
                else:
                    if self.suit_counts[led_suit] > 0:
                        return (self.cards & card_constants.card_suit_masks[led_suit]).astype(np.int8)
                    else:
                        return (self.cards & np.logical_not(card_constants.ace_masks[called_ace])).astype(np.int8)
            elif self.is_picker:
                if self.suit_counts[led_suit] > 0:
                    return self.cards & card_constants.card_suit_masks[led_suit].astype(np.int8)
                elif self.suit_counts[called_ace] > 1:
                    return self.cards
                else:
                    return (self.cards & np.logical_not(card_constants.card_suit_masks[called_ace])).astype(np.int8)
            else:
                if self.suit_counts[led_suit] > 0:
                    return (self.cards & card_constants.card_suit_masks[led_suit]).astype(np.int8)
                else:
                    return self.cards

    # NOTE: This doesn't check for legality
    def play(self, card_int : int):
        self.cards[card_int] = 0
        self.suit_counts[card_constants.card_suits[card_int]] -= 1
        self.number_of_cards -= 1
            


