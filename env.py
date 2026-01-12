from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import functools
from gymnasium.spaces import Sequence, Discrete, MultiBinary, Dict
from gymnasium.utils import seeding
import card_constants
from hand import Hand
from deck import Deck
from suit import Suit
import numpy as np

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    metadata = {"render_modes" : ["human"], "name" : "sheepshead"}

    def __init__(self, render_mode=None):
        # Define possible_agents and render_mode
        self.possible_agents = ["player_" + str(r) for r in range(5)]
        self.render_mode = render_mode

    def _invalid_action(self, agent):
        """
        Whoever does an invalid action pays the table (-12 reward)
        Don't reward other agents in this case.
        """
        for a in self.agents:
            self.rewards[a] = 0
            self.terminations[a] = True
        self.rewards[agent] = -12

    def _normal_agent_step(self):
        if self.agent_selection == "player_0":
            self.agent_selection = "player_1"
        elif self.agent_selection == "player_1":
            self.agent_selection = "player_2"
        elif self.agent_selection == "player_2":
            self.agent_selection = "player_3"
        elif self.agent_selection == "player_3":
            self.agent_selection = "player_4"
        else:
            self.agent_selection = "player_0"

    def _normal_agent_step_sim(self, player) -> str:
        if player == "player_0":
            return "player_1"
        elif player == "player_1":
            return "player_2"
        elif player == "player_2":
            return "player_3"
        elif player == "player_3":
            return "player_4"
        else:
            return "player_0"

    def _get_led_suit(self) -> Suit | None:
        if len(self.current_trick) < 1:
            return None
        else:
            return card_constants.card_suits[self.current_trick[self.leading_player]]

    def _picker_team_won_all_tricks(self) -> bool:
        self.non_picker_agents = self.agents.copy()
        self.non_picker_agents.remove(self.picker)
        if self.partner is not None:
            self.non_picker_agents.remove(self.partner)
        for agent in self.non_picker_agents:
            if self.won_a_trick[agent]:
                return False
        return True

    def _picker_team_won_no_tricks(self):
        if self.partner is None:
            return not self.won_a_trick[self.picker]
        else:
            return not (self.won_a_trick[self.picker] or self.won_a_trick[self.partner])

    def _score_trick(self):
        led_card_idx = self.current_trick[self.leading_player]
        led_suit = self._get_led_suit()
        best_power = card_constants.card_power[led_suit][led_card_idx]
        winning_player = self.leading_player
        for player, card_idx in self.current_trick.items():
            power = card_constants.card_power[led_suit][card_idx]
            if power > best_power:
                best_power = power
                winning_player = player

        if self.partner is None and self.called_ace is not None:
            called_ace_idx = card_constants.aces[self.called_ace]
            if called_ace_idx in self.current_trick.values():
                self.partner = [key for key, val in self.current_trick.items() if val == called_ace_idx][0]

        self.won_a_trick[winning_player] = True
        self.tricks_played += 1
        self.scores[winning_player] += sum([card_constants.card_points[idx] for idx in self.current_trick])
        for _, card_idx in self.current_trick.items():
            self.cards_taken[winning_player][card_idx] = 1
        self.leading_player = winning_player
        self.current_trick = {}

    def _score_game(self):
        if self.is_leaster:
            best_player = None
            is_tie = False
            min_score = 9999
            for agent in self.agents:
                if self.won_a_trick[agent]:
                    if self.scores[agent] < min_score:
                        is_tie = False
                        min_score = self.scores[agent]
                        best_player = agent
                    elif self.scores[agent] == min_score:
                        is_tie = True
            
            if is_tie:
                for agent in self.agents:
                    self.rewards[agent] = 0
            else:
                for agent in self.agents:
                    self.rewards[agent] = -1
                self.rewards[best_player] = 4
        
        elif self.partner is None:
            picker_score = self.scores[self.picker]
            if self._picker_team_won_no_tricks():
               # No tricker against the picker
                for agent in self.agents:
                    self.rewards[agent] = 3
                self.rewards[self.picker] = -12
            elif picker_score < 31:
                # No schneider against the picker
                for agent in self.agents:
                    self.rewards[agent] = 2
                self.rewards[self.picker] = -8
            elif picker_score < 61:
                # Schneider against the picker
                for agent in self.agents:
                    self.rewards[agent] = 1
                self.rewards[self.picker] = -4
            elif picker_score < 91:
                # Schneider for the picker
                for agent in self.agents:
                    self.rewards[agent] = -1
                self.rewards[self.picker] = 4
            elif not self._picker_team_won_all_tricks():
                # No schneider for the picker
                for agent in self.agents:
                    self.rewards[agent] = -2
                self.rewards[self.picker] = 8
            else:
                # No tricker for the picker
                for agent in self.agents:
                    self.rewards[agent] = -3
                self.rewards[self.picker] = 12 
        else:
            picker_team_score = self.scores[self.picker] + self.scores[self.partner]
            if self._picker_team_won_no_tricks():
                # No tricker against the picker team
                for agent in self.agents:
                    self.rewards[agent] = 3
                self.rewards[self.picker] = -6
                self.rewards[self.partner] = -3
            elif picker_team_score < 31:
                # No schneider against the picker team
                for agent in self.agents:
                    self.rewards[agent] = 2
                self.rewards[self.picker] = -4
                self.rewards[self.partner] = -2
            elif picker_team_score < 61:
                # Schneider against the picker team
                for agent in self.agents:
                    self.rewards[agent] = 1
                self.rewards[self.picker] = -2
                self.rewards[self.partner] = -1
            elif picker_team_score < 91:
                # Schneider for the picker team
                for agent in self.agents:
                    self.rewards[agent] = -1
                self.rewards[self.picker] = 2
                self.rewards[self.partner] = 1
            elif not self._picker_team_won_all_tricks():
                # No schneider for the picker team
                for agent in self.agents:
                    self.rewards[agent] = -2
                self.rewards[self.picker] = 4
                self.rewards[self.partner] = 2
            else:
                # No tricker for the picker team
                for agent in self.agents:
                    self.rewards[agent] = -3
                self.rewards[self.picker] = 6
                self.rewards[self.partner] = 3

    # TODO
    def observe(self, agent):
        # Return dictionary with observation and action mask
        """
        Board State:
        - Row 0 is current hand
        - Row 1 is my cards taken
        - Row 2 is opponent 1 cards taken (known)
        - Row 3 is opponent 2 cards taken (known)
        - Row 4 is opponent 3 cards taken (known)
        - Row 5 is opponent 4 cards taken (known)
        - Row 6 is unknown cards

        Current Trick:
        Sequence of card indices

        Called Suit:
        - Just an index for suit
        """
        board_state = np.zeros((7,32), np.int8)
        board_state[0] = self.hands[agent].get_hand_mask()
        if self.picker == agent:
            board_state[1] = self.cards_taken[agent] + self.cards_buried
        else:
            board_state[1] = self.cards_taken[agent]
        opponent1 = self._normal_agent_step_sim(agent)
        board_state[2] = self.cards_taken[opponent1]
        opponent2 = self._normal_agent_step_sim(opponent1)
        board_state[3] = self.cards_taken[opponent2]
        opponent3 = self._normal_agent_step_sim(opponent2)
        board_state[4] = self.cards_taken[opponent3]
        opponent4 = self._normal_agent_step_sim(opponent3)
        board_state[5] = self.cards_taken[opponent4]
        board_state[6] = np.logical_not(np.logical_or(board_state[0], np.logical_or(board_state[1], np.logical_or(board_state[2], np.logical_or(board_state[3], np.logical_or(board_state[4], board_state[5])))))).astype(np.int8)

        trick_as_list = []
        if len(self.current_trick) > 0:
            trick_as_list.append(self.current_trick[self.leading_player])
            curr_player = self.leading_player
            i = 1
            while len(self.current_trick) > i:
                curr_player = self._normal_agent_step_sim(curr_player)
                trick_as_list.append(self.current_trick[curr_player])
                i += 1
        current_trick = tuple(trick_as_list)

        if self.called_ace == Suit.CLUB:
            called_suit = 0
        elif self.called_ace == Suit.SPADE:
            called_suit = 1
        elif self.called_ace == Suit.HEART:
            called_suit = 2
        else:
            called_suit = 3

        observation = {}
        observation["Board State"] = board_state
        observation["Current Trick"] = current_trick
        observation["Called Suit"] = called_suit
 
    def close(self):
        # End extra displays
        pass

    def reset(self, seed=None, options=None):
        """
        Initialize the following:
            - agents
            - rewards
            - _cumulative_rewards
            - terminations
            - truncations
            - infos
            - agent_selection
        """
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}


        self.initial_player = self.np_random.choice(self.agents)
        self.agent_selection = self.initial_player
        self.leading_player = self.initial_player

        self.deck = Deck(self.np_random)
        self.deck.shuffle() 
        result = self.deck.deal()
        self.hands : dict[str, Hand] = {agent: Hand() for agent in self.agents}
        for i in range(5):
            self.hands[f"player_{i}"].add_cards(result[f"hand{i+1}"].tolist())
        self.cards_taken = {agent : np.zeros(32, dtype=np.int8) for agent in self.agents}
        self.cards_buried = np.zeros(32, dtype=np.int8)
        self.scores = {agent: 0 for agent in self.agents}
        self.won_a_trick = {agent: False for agent in self.agents}
        self.called_ace : Suit = None
        self.game_phase = "BIDDING"
        self.current_trick = {}
        self.passed = 0
        self.is_leaster = False
        self.picker = None
        self.partner = None
        self.blind = result["blind"]
        self.leaster_blind_points = 0
        self.tricks_played = 0

    # TODO May need to be increased
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Return observation space
        return Dict(spaces={
            "Board State" : MultiBinary([7,32]),
            "Current Trick" : Sequence(Discrete(32)),
            "Called Suit" : Discrete(4)
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Return action space
        return Discrete(70)

    def step(self, action):
        """
        Call _was_dead_step() if the current agent is already terminated or truncated
        - Take in an action and update the following:
            - rewards
            - _cumulative_rewards
            - terminations
            - truncations
            - infos
            - agent_selection
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        if self.game_phase == "BIDDING":
            """
            Only allow Action 0 or 1

            Action 0 - Pass
            Action 1 - Pick
            """
            if action == 0:
                # Pass
                self.passed += 1
                if self.passed == 5:
                    self.agent_selection = self.initial_player
                    self.is_leaster = True
                    for card_idx in self.blind:
                        self.leaster_blind_points += card_constants.card_points[card_idx]
                    self.game_phase == "STANDARD"
                else:
                    self._normal_agent_step()
            elif action == 1:
                # Pick
                self.picker = agent
                self.hands[agent].add_cards(self.blind)
                self.game_phase = "PARTNER_DECLARE"
            else:
                # Invalid action, everyone else wins
                self._invalid_action(agent)

        elif self.game_phase == "PARTNER_DECLARE":
            """
            Allow actions 2-5. Only the picker agent gets here
            
            Action 2 - Declare "Go Alone"
            Action 3 - Declare "Ace of Clubs"
            Action 4 - Declare "Ace of Spades"
            Action 5 - Declare "Ace of Hearts"

            """
            if action == 2:
                # Go alone
                self.called_ace = None
                self.game_phase = "BURYING"
            elif action == 3:
                # Call Ace of Clubs
                self.called_ace = Suit.CLUB
                self.game_phase = "BURYING"
            elif action == 4:
                # Call Ace of Spades
                self.called_ace = Suit.SPADE
                self.game_phase = "BURYING"
            elif action == 5:
                # Call Ace of Hearts
                self.called_ace = Suit.HEART
                self.game_phase = "BURYING"
            else:
                # Invalid action
                self._invalid_action(agent)

        elif self.game_phase == "BURYING":
            """
            Allow actions 6-37. Only the picker agent gets here
            """
            if action >= 6 and action <= 37:
                card_idx = action - 6
                if self.hands[agent].can_bury_card(card_idx, self.called_ace):
                    self.scores[agent] += self.hands[agent].bury(card_idx)
                    self.cards_buried[card_idx] = 1
                    if self.hands[agent].number_of_cards == 6:
                        self.game_phase = "STANDARD"
                        self.agent_selection = self.initial_player
                    else:
                        self.game_phase = "BURYING"
                        self.agent_selection = agent
                else:
                    # Invalid action
                    self._invalid_action(agent)
            else:
                # Invalid action
                self._invalid_action(agent)

        elif self.game_phase == "STANDARD":
            """
            Allow actions 38-69
            """
            if action >= 38 and action <= 69:
                card_idx = action - 38
                if self.hands[agent].can_play_card(card_idx, self._get_led_suit(), self.called_ace):
                    self.hands[agent].play(card_idx)
                    self.current_trick[agent] = card_idx
                    if len(self.current_trick) == 5:
                        self._score_trick()
                        if self.tricks_played == 6:
                            self._score_game()
                            for a in self.agents:
                                self.terminations[a] = True
                else:
                    self._invalid_action(agent)
            else:
                self._invalid_action(agent)

        self._accumulate_rewards()

