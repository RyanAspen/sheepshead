"""
Class which handles details about how cards are played

TODO: Logic is wrong for numpy indicing!!

"""

from typing import Optional
import numpy as np

class CardManager:
    def __init__(self):
        self.card_ids = ["7H","7D","7C","7S","8H","8D","8C","8S","9H","9D","9C","9S","KH","KD","KC","KS","10H","10D","10C","10S","AH","AD","AC","AS","JH","JD","JC","JS","QH","QD","QC","QS"]
        
        # 0 = club, 1 = spade, 2 = heart, 3 = trump
        self.card_suits = np.array([2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,3,3,3,3,3,3,3,3])
        
        # 0 = 7, 1 = 8, 2 = 9, 3 = K, 4 = 10, 5 = A, 6 = J, 7 = Q
        self.card_ranks = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7])

        self.card_points = np.array([0,0,0,0,0,0,0,0,0,0,0,0,4,4,4,4,10,10,10,10,11,11,11,11,2,2,2,2,3,3,3,3])

        self.called_ace_idx = np.array([22,23,20,21])

    def get_card_id(self, card : int) -> str:
        return self.card_ids[card]

    """
    Given a hand, determine what cards can legally be played
    Partner Rules:
    - If we are the picker, we called an ace, and the ace's suit was led, we must play a card in the ace's suit. If we do not have any of those cards, we must instead play the "unknown" card.
    - If we are the picker, we called an ace, and the ace's suit was not led, we must leave at least one card in the ace's suit in our hand unless it is our last card in our hand. If we do not have any of those cards, we must instead save the "unknown" card.
    - If we are the partner and the ace's suit was led, we must play the ace.
    - If we are the partner and the ace's suit was not led, we cannot play the ace unless it is our last card in our hand.
    """
    def get_legal_cards(self, hand : np.ndarray, is_picker : bool, partner_known : bool, called_ace : Optional[int] = None, led_card : Optional[int] = None) -> np.ndarray:
        """
        - If we only have one card, play it.
        - If we are leading, we can play anything as long as it doesn't violate partner rules
        - If we have the called ace and that suit is led, we must play that ace
        - Limit legal cards to the led suit without the called ace. If there is at least one legal card, play one.
        - If we are the picker and the called ace has not been played, we cannot play a card from
        the called ace's suit if we would then have no cards of the ace's suit
        - If we have the called ace, do not play it.
        """
        
        # - If we only have one card, play it.
        if hand.size == 1:
            return hand
        
        if called_ace is None:
            called_ace_idx = -1
        else:
            called_ace_idx = self.called_ace_idx[called_ace]

        # - If we are leading, we can play anything as long as it doesn't violate partner rules
        if led_card is None:
            """
            - We cannot play the called ace
            - If we are the picker, we called a partner, and we only have one card of the called suit, we cannot play it
            - Every other card is fair game
            """
            if called_ace is None:
                return hand
            else:
                hand = hand[hand != called_ace_idx]
                if is_picker:
                    called_suit_cards = hand[self.card_suits[hand] == called_ace]
                    if called_suit_cards.size == 1:
                        hand = hand[self.card_suits[hand] != called_ace]
                return hand
        
        # - If we have the called ace and that suit is led, we must play that ace
        led_suit = self.card_suits[led_card]
        if called_ace_idx in hand and led_suit == called_ace:
            return np.array([called_ace_idx])
        
        # - Limit legal cards to the led suit without the called ace. If there is at least one legal card, play one.
        cards_following_suite = hand[self.card_suits[hand] == led_suit]
        if cards_following_suite.size > 0:
            return cards_following_suite
        
        # - If we are the picker and the called ace has not been played, we cannot play a card from
        # the called ace's suit if we would then have no cards of the ace's suit

        if is_picker and not partner_known:
            if hand[self.card_suits[hand] == called_ace].size == 1:
                hand = hand[self.card_suits[hand] != called_ace]

        # Don't play the called ace
        return hand[hand != called_ace_idx]

    def is_card_better(self, current_card : int, new_card : int, led_card : int) -> bool:
        """
        Rules are as follows:
        1) Trump cards beat non-trump cards
        2) Cards in the current suit beat cards that are not
        3) Cards with higher rank win
        4) Jack and Queen cards break ties with the order QC > QS > QH > QD > JC > JS > JH > JD
        """
        current_suit = self.card_suits[current_card]
        new_suit = self.card_suits[new_card]

        if current_suit == 3 and new_suit != 3:
            return False
        elif current_suit != 3 and new_suit == 3:
            return True
        elif current_suit == 3:
            # Both cards are trump
            current_rank = self.card_ranks[current_card]
            new_rank = self.card_ranks[new_card]
            if current_rank > new_rank:
                return False
            elif current_rank < new_rank:
                return True
            else:
                # Must be both jacks or queens. Use tiebreaker
                if current_suit < new_suit:
                    return False
                else:
                    return True
        else:
            # Both cards are fail suits, check led_suit
            led_suit = self.card_suits[led_card]
            if current_suit == led_suit and new_suit != led_suit:
                return False
            elif current_suit != led_suit and new_suit == led_suit:
                return True
            elif current_suit != led_suit:
                # If neither card follows suit, the comparison is meaningless. Choose True arbitrarily
                return True
            else:
                # Use rank
                current_rank = self.card_ranks[current_card]
                new_rank = self.card_ranks[new_card]
                if current_rank > new_rank:
                    return False
                else:
                    return True

    def get_winner(self, trick : np.ndarray) -> int:
        led_card = trick[0]
        best_card = trick[0]
        best_idx = 0
        for i in range(len(trick)-1):
            if self.is_card_better(trick[i+1], best_card, led_card):
                best_card = trick[i+1]
                best_idx = i+1
        return best_idx
    
    def get_trick_score(self, trick : np.ndarray) -> int:
        return self.card_points[trick].sum()
