from pypokerengine.players import BasePokerPlayer
from engine.state_encoder import encode_state
import torch
import numpy as np
import random
import copy

class DeepCFRPlayer(BasePokerPlayer):
    """A poker AI player that uses Deep CFR (deep neural networks for regret and strategy)."""
    # Define a consistent order for actions: index 0->'raise', 1->'call', 2->'fold'
    ACTIONS_ORDER = ['raise', 'call', 'fold']

    def __init__(self, name, regret_net, strategy_net):
        super(DeepCFRPlayer, self).__init__()
        self.name = name
        self.uuid = None  # will be set once game starts
        self.regret_net = regret_net
        self.strategy_net = strategy_net
        self.episode_history = []  # to record states, strategies, and actions for this episode
        self.my_hole = None        # store hole cards for this player at round start
        self.simulating = False    # flag to indicate if the player is in simulation mode (no logging)

    def receive_game_start_message(self, game_info):
        """Called once at the beginning of the game. We capture our UUID and game settings."""
        # Identify our player's UUID from game_info using our registered name
        for player in game_info['seats']:
            if player['name'] == self.name:
                self.uuid = player['uuid']
                break
        # (Optional) store game rules if needed from game_info['rule']

    def receive_round_start_message(self, round_count, hole_card, seats):
        """Called at the start of a new round (hand). We store our hole cards."""
        self.my_hole = hole_card[:]  # copy list of hole card strings
        # Clear any previous round data
        self.episode_history = []

    def declare_action(self, valid_actions, hole_card, round_state):
        """Decide an action (fold, call, or raise) based on regret network output and regret matching."""
        # Encode the current state from our perspective
        state_vec = encode_state(self.uuid, hole_card, round_state, valid_actions)
        # Convert to torch tensor for network input
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)  # add batch dimension
        # Get regret values for actions from the regret network
        with torch.no_grad():
            regret_out = self.regret_net(state_tensor)
        # Convert to numpy array
        regret_values = regret_out.cpu().numpy().flatten()
        # Map regrets to actions in defined order
        regrets = {action: regret_values[idx] for idx, action in enumerate(DeepCFRPlayer.ACTIONS_ORDER)}
        # Determine which actions are allowed in this state
        allowed_actions = [act["action"] for act in valid_actions]
        # Apply regret matching: use positive part of regrets
        strategy_dist = {}
        sum_positive = 0.0
        for action in allowed_actions:
            r = regrets.get(action, 0.0)
            if r < 0:
                r = 0.0
            strategy_dist[action] = r
            sum_positive += r
        if sum_positive <= 1e-9:
            # If no positive regret, use a uniform random strategy among allowed actions
            for action in allowed_actions:
                strategy_dist[action] = 1.0 / len(allowed_actions)
        else:
            # Normalize positive regrets to probabilities
            for action in allowed_actions:
                strategy_dist[action] = strategy_dist[action] / sum_positive
        # Construct full 3-length strategy vector for logging (zero for disallowed actions)
        strategy_vec = np.zeros(len(DeepCFRPlayer.ACTIONS_ORDER), dtype=np.float32)
        for action, prob in strategy_dist.items():
            idx = DeepCFRPlayer.ACTIONS_ORDER.index(action)
            strategy_vec[idx] = prob
        # Choose an action according to the probabilities (stochastic policy)
        actions_list = list(strategy_dist.keys())
        probs_list = [strategy_dist[a] for a in actions_list]
        # print(probs_list)
        chosen_action = random.choices(actions_list, weights=probs_list, k=1)[0]
        # Determine the amount for the chosen action
        if chosen_action == 'fold':
            chosen_amount = 0
        elif chosen_action == 'call':
            call_info = next(act for act in valid_actions if act["action"] == "call")
            chosen_amount = call_info["amount"]
        elif chosen_action == 'raise':
            # Use the maximum raise amount (e.g., all-in) for raising
            raise_info = next(act for act in valid_actions if act["action"] == "raise")
            chosen_amount = raise_info["amount"]
        # Log the decision if not in a simulation (real gameplay)
        if not self.simulating:
            state_snapshot = copy.deepcopy(round_state)  # store state for regret simulation
            self.episode_history.append({
                "player_uuid": self.uuid,
                "state_vec": state_vec,
                "strategy": strategy_vec,
                "action": chosen_action,
                "raise_amount": chosen_amount,
                "valid_actions": allowed_actions,
                "valid_actions_detail": copy.deepcopy(valid_actions),
                "round_state": state_snapshot,
                "hole_card": hole_card[:]  # store hole cards for this decision point
            })
        # print("== DECLARE_ACTION ==")
        # print("Valid actions:", valid_actions)
        # print("Hole cards:", hole_card)
        # print("action: ", chosen_action)
        # print("raise_amount: ", chosen_amount)
        # print("Round state:", round_state.get('street'))
        return chosen_action, chosen_amount

    def receive_street_start_message(self, street, round_state):
        pass  # or leave empty if you don't need to handle it

    def receive_game_update_message(self, action, round_state):
        """Called during the game when an action is performed. Not used for our strategy."""
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        """Called at the end of a round for cleanup or logging."""
        pass