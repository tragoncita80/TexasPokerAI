from pypokerengine.api.game import setup_config, start_poker
from ai.deep_cfr_player import DeepCFRPlayer

class CFRRunner:
    def __init__(self, regret_net, strategy_net, initial_stack=1000, small_blind=5, ante=0):
        """
        Initialize the self-play runner with game parameters and shared networks.
        """
        self.regret_net = regret_net
        self.strategy_net = strategy_net
        self.initial_stack = initial_stack
        self.small_blind = small_blind
        self.ante = ante
        self.blind_structure = {}  # default (no blind escalation)

    def play_episode(self):
        """
        Simulate one poker game (one hand) between two DeepCFR players.
        Returns:
            players: [player1, player2] DeepCFRPlayer instances after the game.
            payoffs: dict mapping each player's UUID to their payoff for the round.
        """
        # Create two players with shared networks
        player1 = DeepCFRPlayer(name="Player1", regret_net=self.regret_net, strategy_net=self.strategy_net)
        player2 = DeepCFRPlayer(name="Player2", regret_net=self.regret_net, strategy_net=self.strategy_net)
        # Configure a single round of heads-up poker
        config = setup_config(max_round=1, initial_stack=self.initial_stack, 
                               small_blind_amount=self.small_blind)
        config.register_player(name="Player1", algorithm=player1)
        config.register_player(name="Player2", algorithm=player2)
        # Run the round (start_poker returns after one hand since max_round=1)
        game_result = start_poker(config, verbose=0)
        # Compute payoffs: final stack minus initial stack for each player
        payoffs = {}
        for player_info in game_result['players']:
            final_stack = player_info['stack']
            payoffs[player_info['uuid']] = final_stack - self.initial_stack
        return [player1, player2], payoffs