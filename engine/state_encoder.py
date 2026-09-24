import numpy as np

# Define card rank and suit order for encoding
RANKS = "23456789TJQKA"
SUITS = "CDHS"  # Clubs, Diamonds, Hearts, Spades

rank_to_index = {rank: i for i, rank in enumerate(RANKS)}
suit_to_index = {suit: i for i, suit in enumerate(SUITS)}

def card_to_index(card_str):
    """
    Converts a card string (e.g. 'H7' for 7 of Hearts, 'DA' for Ace of Diamonds)
    into a 0-51 index for one-hot encoding.
    """
    # Format: first char is suit (C, D, H, S), remaining char(s) is rank (2-9, T, J, Q, K, A).
    suit = card_str[0]
    rank = card_str[1]
    return 13 * suit_to_index[suit] + rank_to_index[rank]

def encode_state(player_uuid, hole_cards, round_state, valid_actions_detail):
    """
    Encodes the current game state and player's private cards into a fixed-length vector.
    - player_uuid (str): UUID of the current player (whose perspective we encode).
    - hole_cards (list): Player's hole cards as list of strings, e.g. ['H7', 'DA'].
    - round_state (dict): Current public round state from PyPokerEngine.
    Returns:
        np.ndarray: 1D float array representing the encoded state.
    """
    # One-hot encode hole cards (52-length vector)
    initial = 1000
    hole_vec = np.zeros(52, dtype=np.float32)
    for card in hole_cards:
        idx = card_to_index(card)
        hole_vec[idx] = 1.0
    # One-hot encode community cards (52-length vector)
    board_vec = np.zeros(52, dtype=np.float32)
    community_cards = round_state.get('community_card', [])
    for card in community_cards:
        idx = card_to_index(card)
        board_vec[idx] = 1.0
    # One-hot encode the current street (preflop, flop, turn, river)
    street_list = ['preflop', 'flop', 'turn', 'river']
    street_vec = np.zeros(len(street_list), dtype=np.float32)
    street = round_state.get('street', 'preflop').lower()
    if street in street_list:
        street_idx = street_list.index(street)
        street_vec[street_idx] = 1.0
    # Numeric features: pot size, call amount (if any), player stacks, and dealer indicator
    pot = round_state['pot']['main']['amount'] / initial
    call_amount = 0.0  # Default 0 (if no call needed)
    # We derive my_stack and opp_stack from round_state['seats']
    my_stack = 0.0
    opp_stack = 0.0
    for act in valid_actions_detail:
        if act['action'] == 'call':
            call_amount = float(act['amount']) / initial
        elif act['action'] == 'raise':
            raise_amount = float(act['amount']) / initial
    seats = round_state.get('seats')
    if seats:
        for seat in seats:
            if seat['uuid'] == player_uuid:
                my_stack = float(seat['stack']) / initial
            else:
                opp_stack = float(seat.get('stack', 0)) / initial
    numeric_features = np.array([my_stack, opp_stack, float(pot), float(call_amount), float(raise_amount)], dtype=np.float32)
    # print(numeric_features)
    # Concatenate all parts: hole cards, board cards, street, numeric features
    state_vector = np.concatenate([hole_vec, board_vec, street_vec, numeric_features])
    return state_vector