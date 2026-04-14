import numpy as np
from numpy.typing import NDArray
from typing import Dict, List
from sheepshead.suit import Suit
import pygame

card_ids : List[str] = ["7H","7D","7C","7S","8H","8D","8C","8S","9H","9D","9C","9S","KH","KD","KC","KS","10H","10D","10C","10S","AH","AD","AC","AS","JH","JD","JC","JS","QH","QD","QC","QS"]
card_image_paths : List[str] = [
    "7_of_hearts.png",
    "7_of_diamonds.png",
    "7_of_clubs.png",
    "7_of_spades.png",
    "8_of_hearts.png",
    "8_of_diamonds.png",
    "8_of_clubs.png",
    "8_of_spades.png",
    "9_of_hearts.png",
    "9_of_diamonds.png",
    "9_of_clubs.png",
    "9_of_spades.png",
    "king_of_hearts.png",
    "king_of_diamonds.png",
    "king_of_clubs.png",
    "king_of_spades.png",
    "10_of_hearts.png",
    "10_of_diamonds.png",
    "10_of_clubs.png",
    "10_of_spades.png",
    "ace_of_hearts.png",
    "ace_of_diamonds.png",
    "ace_of_clubs.png",
    "ace_of_spades.png",
    "jack_of_hearts.png",
    "jack_of_diamonds.png",
    "jack_of_clubs.png",
    "jack_of_spades.png",
    "queen_of_hearts.png",
    "queen_of_diamonds.png",
    "queen_of_clubs.png",
    "queen_of_spades.png",
]
card_suits : List[Suit] = [
    Suit.HEART,
    Suit.TRUMP, 
    Suit.CLUB, 
    Suit.SPADE,
    Suit.HEART,
    Suit.TRUMP, 
    Suit.CLUB, 
    Suit.SPADE,
    Suit.HEART,
    Suit.TRUMP, 
    Suit.CLUB, 
    Suit.SPADE,
    Suit.HEART,
    Suit.TRUMP, 
    Suit.CLUB, 
    Suit.SPADE,
    Suit.HEART,
    Suit.TRUMP, 
    Suit.CLUB, 
    Suit.SPADE,
    Suit.HEART,
    Suit.TRUMP, 
    Suit.CLUB, 
    Suit.SPADE,
    Suit.TRUMP,
    Suit.TRUMP, 
    Suit.TRUMP, 
    Suit.TRUMP, 
    Suit.TRUMP,
    Suit.TRUMP, 
    Suit.TRUMP, 
    Suit.TRUMP
]

card_suit_masks : Dict[Suit, NDArray[np.bool]] = {
    Suit.CLUB : np.array([0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0], dtype=np.bool),
    Suit.SPADE : np.array([0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0], dtype=np.bool),
    Suit.HEART : np.array([1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], dtype=np.bool),
    Suit.TRUMP : np.array([0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,1,1,1,1,1,1,1], dtype=np.bool)
}

card_points : List[int] = [0,0,0,0,0,0,0,0,0,0,0,0,4,4,4,4,10,10,10,10,11,11,11,11,2,2,2,2,3,3,3,3]

card_power : Dict[Suit, list[int]] = {
    Suit.CLUB : [0,7,1,0,0,8,2,0,0,9,3,0,0,10,4,0,0,11,5,0,0,12,6,0,14,13,16,15,18,17,20,19],
    Suit.SPADE : [0,7,0,1,0,8,0,2,0,9,0,3,0,10,0,4,0,11,0,5,0,12,0,6,14,13,16,15,18,17,20,19],
    Suit.HEART : [1,7,0,0,2,8,0,0,3,9,0,0,4,10,0,0,5,11,0,0,6,12,0,0,14,13,16,15,18,17,20,19],
    Suit.TRUMP : [0,7,0,0,0,8,0,0,0,9,0,0,0,10,0,0,0,11,0,0,0,12,0,0,14,13,16,15,18,17,20,19]
}

aces : Dict[Suit, int] = {
    Suit.CLUB : 22,
    Suit.SPADE : 23,
    Suit.HEART : 20,
    Suit.TRUMP : 21
}

ace_masks : Dict[Suit, NDArray[np.bool]] = {
    Suit.CLUB : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], dtype=np.bool),
    Suit.SPADE : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], dtype=np.bool),
    Suit.HEART : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], dtype=np.bool),
    Suit.TRUMP : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], dtype=np.bool)
}