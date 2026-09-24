import torch
import random
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from ai.strategy_net import StrategyNet
from engine.state_encoder import encode_state
from players.strategy_player import StrategyPlayer
from players.random_player import RandomPlayer
from players.fish_player import FishPlayer
from players.console_player import ConsolePlayer
from tqdm import tqdm

# Load the trained strategy_net
strategy_net = StrategyNet(input_dim=113)
strategy_net.load_state_dict(torch.load("models/strategy_net.pt"))
strategy_net.eval()

# Configure game
config = setup_config(max_round=100, initial_stack=1000, small_blind_amount=10)
config.register_player(name="StrategyAI", algorithm=StrategyPlayer(strategy_net))
# config.register_player(name="RandomAI", algorithm=RandomPlayer())
# config.register_player(name="Console", algorithm=ConsolePlayer())
config.register_player(name="Fish1", algorithm=FishPlayer())

# Start the game
result = start_poker(config, verbose=1)

print("\n=== Game Result ===")
for player in result["players"]:
    print(f"{player['name']} - Final Stack: {player['stack']}")

strategy_full_wins = 0
total_matches = 10000
initial_stack = 1000
small_blind = 10

for match in tqdm(range(total_matches)):
    config = setup_config(max_round=1000000000, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="AI", algorithm=StrategyPlayer(strategy_net))
    config.register_player(name="Fish", algorithm=FishPlayer())

    result = start_poker(config, verbose=0)

    # print(f"match {match+1} result:")
    for player in result["players"]:
        if player['name']=='AI' and player['stack']>0: strategy_full_wins+=1
        # print(f"{player['name']} - Final Stack: {player['stack']}")

# Print summary
print("\n=== Match Summary ===")
print(f"Total matches: {total_matches}")
print(f"StrategyAI full wins (opponent busted): {strategy_full_wins}")