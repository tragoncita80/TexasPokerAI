import random
import numpy as np
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card, attach_hole_card_from_deck
from pypokerengine.utils.card_utils import gen_cards
from ai.deep_cfr_player import DeepCFRPlayer

def compute_counterfactual_regrets(players, payoffs, initial_stack, small_blind, ante=0, blind_structure=None):
    """
    Compute counterfactual regrets for all decision points in a self-play episode.
    - players: list of DeepCFRPlayer objects from the episode.
    - payoffs: dict mapping each player UUID to their actual payoff (final stack - initial stack).
    - initial_stack, small_blind, ante: game parameters for emulator.
    - blind_structure: blind structure dict (if any).
    Returns: list of (state_vector, regret_vector) samples for training.
    """
    regret_samples = []
    # Enable simulation mode to prevent logging new data during regret computation
    for p in players:
        p.simulating = True
    # Iterate through each decision in each player's history
    for player in players:
        for decision in player.episode_history:
            player_uuid = decision["player_uuid"]
            actual_action = decision["action"]
            allowed_actions = decision["valid_actions"]
            round_state = decision["round_state"]
            hole_cards = decision["hole_card"]
            actual_payoff = payoffs.get(player_uuid, 0)
            # Prepare a regret vector for [raise, call, fold]
            regret_vec = np.zeros(len(DeepCFRPlayer.ACTIONS_ORDER), dtype=np.float32)
            # Simulate each alternative action that was available at this decision
            for alt_action in allowed_actions:
                if alt_action == actual_action:
                    continue  # skip the action actually taken
                # Determine the bet amount for the alternate action
                if alt_action == "fold":
                    alt_amount = 0
                elif alt_action == "call":
                    call_info = next(act for act in decision["valid_actions_detail"] if act["action"] == "call")
                    alt_amount = call_info["amount"]
                elif alt_action == "raise":
                    raise_info = next(act for act in decision["valid_actions_detail"] if act["action"] == "raise")
                    alt_amount = raise_info["amount"]
                # Restore the game state at the decision point
                game_state = restore_game_state(round_state)
                # Attach known hole cards for each player to the game state
                for pl in game_state["table"].seats.players:
                    if pl.uuid == player_uuid:
                        game_state = attach_hole_card(game_state, pl.uuid, gen_cards(hole_cards))
                    else:
                        opp_player = next(obj for obj in players if obj.uuid == pl.uuid)
                        opp_hole = opp_player.my_hole
                        game_state = attach_hole_card(game_state, pl.uuid, gen_cards(opp_hole))
                # Initialize emulator and register players (using their current strategy behavior)
                emulator = Emulator()
                emulator.set_game_rule(len(players), max_round=1, small_blind_amount=small_blind, ante_amount=ante)
                if blind_structure:
                    emulator.set_blind_structure(blind_structure)
                for pl in players:
                    emulator.register_player(pl.uuid, pl)
                # Apply the alternate action and play out the rest of the hand
                updated_state, ev1 = emulator.apply_action(game_state, alt_action, alt_amount)
                final_state, ev2 = emulator.run_until_game_finish(updated_state)
                events = ev1 + ev2
                # Get final stacks from the game finish event
                final_stacks = {}
                for event in events:
                    if event["type"] == "event_game_finish":
                        for pl_info in event["players"]:
                            final_stacks[pl_info["uuid"]] = pl_info["stack"]
                        break
                # Compute the payoff for this player under the alternate action
                alt_final_stack = final_stacks.get(player_uuid, 0)
                alt_payoff = alt_final_stack - initial_stack
                # Regret is the difference in payoff compared to the actual outcome
                regret_value = alt_payoff - actual_payoff
                action_index = DeepCFRPlayer.ACTIONS_ORDER.index(alt_action)
                regret_vec[action_index] = float(regret_value)
            # Ensure the chosen action's regret is 0
            chosen_idx = DeepCFRPlayer.ACTIONS_ORDER.index(actual_action)
            regret_vec[chosen_idx] = 0.0
            # Save the sample (state vector and regret targets)
            state_vector = decision["state_vec"]
            regret_samples.append((state_vector, regret_vec))
    # Disable simulation mode after calculations
    for p in players:
        p.simulating = False
    return regret_samples