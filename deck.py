"""
Class which manages the deck.
"""

import numpy as np

class Deck:
    def __init__(self):
        # 32 cards in a sheepshead deck
        self.rng = np.random.default_rng()
        self.deck = np.arange(len(self.card_ids))
    
    def shuffle(self):
        self.rng.shuffle(self.deck)

    # Assuming standard 5-player game
    def deal(self) -> dict[str, np.ndarray]:
        hand1_indices = [0,1,11,12,22,23]
        hand2_indices = [2,3,13,14,24,25]
        hand3_indices = [4,5,15,16,26,27]
        hand4_indices = [6,7,17,18,28,29]
        hand5_indices = [8,9,19,20,30,31]
        blind_indices = [10,21]

        result = {}

        result["hand1"] = self.deck[hand1_indices]
        result["hand2"] = self.deck[hand2_indices]
        result["hand3"] = self.deck[hand3_indices]
        result["hand4"] = self.deck[hand4_indices]
        result["hand5"] = self.deck[hand5_indices]
        result["blind"] = self.deck[blind_indices]

        return result
