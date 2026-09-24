# Define a simple random opponent
from pypokerengine.players import BasePokerPlayer
import random

class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        action = random.choice(valid_actions)
        act_name = action["action"]
        amt = action["amount"]
        if act_name == "raise":
            if isinstance(amt, dict):
                min_raise = int(amt["min"])
                max_raise = int(amt["max"])
                if min_raise >= max_raise:
                    amt = min_raise
                else:
                    amt = random.randint(min_raise, max_raise)
            else:
                amt = int(amt)
        else:
            amt = int(amt)
        return act_name, amt

    def receive_game_start_message(self, game_info):pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return RandomPlayer()