import random
import torch
from engine.state_encoder import encode_state
from pypokerengine.players import BasePokerPlayer
from ai.strategy_net import StrategyNet
import numpy as np

class StrategyPlayer(BasePokerPlayer):
    def __init__(self, model, name="StrategyAI"):
        self.model = model
        self.name = name
        self.uuid = None
    
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass
    def receive_game_start_message(self, game_info):
        for p in game_info["seats"]:
            if p["name"] == self.name:
                self.uuid = p["uuid"]
                break

    def declare_action(self, valid_actions, hole_card, round_state):
        state_vec = encode_state(self.uuid, hole_card, round_state, valid_actions)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        probs = self.model(state_tensor).detach().numpy()[0]

        legal_actions = [a["action"] for a in valid_actions]
        all_actions = ["raise", "call", "fold"]

        masked = np.array([probs[i] if act in legal_actions else 0.0
                           for i, act in enumerate(all_actions)], dtype=np.float32)
        total = masked.sum()
        if total <= 0.0:
            masked = np.array([1.0 if act in legal_actions else 0.0 for act in all_actions], dtype=np.float32)
            total = masked.sum()
        masked /= total

        choice = np.random.choice(all_actions, p=masked)
        # print(probs)
        # print(choice)
        if choice == "fold":
            return "fold", 0
        elif choice == "call":
            return "call", next(a["amount"] for a in valid_actions if a["action"]=="call")
        elif choice == "raise":
            return "raise", next(a["amount"] for a in valid_actions if a["action"]=="raise")

def setup_ai():
    # Load the trained strategy_net
    strategy_net = StrategyNet(input_dim=113)
    strategy_net.load_state_dict(torch.load("models/strategy_net.pt"))
    strategy_net.eval()
    return StrategyPlayer(strategy_net)