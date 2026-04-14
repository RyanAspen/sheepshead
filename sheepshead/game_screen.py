import pygame
import sheepshead.card_constants as card_constants
import pathlib 

SCREEN_SIZE = (1000,1000)
CARD_SIZE = (150,100)
TRICK_CARDS_Y = 300
HAND_CARDS_Y = 600
BACKGROUND_COLOR = (0,255,0)
RENDER_FPS = 5

class GameScreen:
    def __init__(self, render_mode : str):
        self.screen : pygame.Surface = None
        self.render_mode : str = render_mode

        def get_image(card_int : int):
            image_path = pathlib.Path("sheepshead") / "card_images" / card_constants.card_image_paths[card_int]
            image = pygame.image.load(image_path)
            image = pygame.transform.scale(image, CARD_SIZE)
            return image

        self.card_images : list[pygame.Surface] = [get_image(i) for i in range(32)]
        self.clock : pygame.time.Clock = pygame.time.Clock()

    def reset(self):
        if self.render_mode == "human" and self.screen is None:
            pygame.init()
        
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(SCREEN_SIZE)
            pygame.display.set_caption("Sheepshead")
        elif self.render_mode == "rgb_array":
            self.screen = pygame.Surface(SCREEN_SIZE)

    def render(self, cards_this_trick : list[int], cards_in_hand : list[int]):
        self.screen.fill(BACKGROUND_COLOR)
        
        # Rendering played cards
        if len(cards_this_trick) > 0:
            start_x = int((SCREEN_SIZE[0] - (len(cards_this_trick) * CARD_SIZE[0])) / 2)
            for card_int in cards_this_trick:
                self.screen.blit(self.card_images[card_int], (start_x, TRICK_CARDS_Y))
                start_x += CARD_SIZE[0]

        # Rendering cards in hand
        if len(cards_in_hand) > 0:
            start_x = int((SCREEN_SIZE[0] - (len(cards_in_hand) * CARD_SIZE[0])) / 2)
            for card_int in cards_in_hand:
                self.screen.blit(self.card_images[card_int], (start_x, HAND_CARDS_Y))
                start_x += CARD_SIZE[0]

        pygame.display.update()
        self.clock.tick(RENDER_FPS)

    def print(self, current_player : str, cards_this_trick : list[int], cards_in_hand : list[int]):
        print(f"Current Player: {current_player}")
        cards_this_trick_str = ",".join([card_constants.card_ids[c] for c in cards_this_trick])
        print(f"Cards played this trick: {cards_this_trick_str}")
        cards_in_hand_str = ",".join([card_constants.card_ids[c] for c in cards_in_hand])
        print(f"Cards in hand: {cards_in_hand_str}")
