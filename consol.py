#!/usr/bin/env python3
import sys
import torch
import numpy as np
from ai.strategy_net import StrategyNet
from engine.state_encoder import encode_state

SUITS = {"C", "D", "H", "S"}
RANKS = set("23456789TJQKA")
COMMUNITY_COUNTS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
ALL_ACTIONS = ["raise", "call", "fold"]

def is_valid_card(tok):
    return len(tok) == 2 and tok[0] in SUITS and tok[1] in RANKS

def build_round_state_and_actions(street, hole, comm, sb=10, init_stack=1000):
    seats = [{"uuid":"me","stack":init_stack},{"uuid":"opp","stack":init_stack}]
    pot = sb + sb*2
    rs = {"street":street, "community_card":comm, "pot":{"main":{"amount":pot}}, "seats":seats}
    if street=="preflop":
        call_amt = sb*2 - sb
        raise_amt = sb*2
    else:
        call_amt = sb*2
        raise_amt = sb*2
    va = [{"action":"fold","amount":0},
          {"action":"call","amount":call_amt},
          {"action":"raise","amount":raise_amt}]
    return rs, va

def print_probs(raw, probs, choice):
    print("\n Raw logits:")
    for a,p in zip(ALL_ACTIONS, raw):  print(f"  {a:5s}: {p:.4f}")
    print(" Masked probs:")
    for a,p in zip(ALL_ACTIONS, probs): print(f"  {a:5s}: {p:.4f}")
    print(f"\n → Suggest: {choice}\n" + "-"*40)

def main():
    model = StrategyNet(input_dim=113)
    model.load_state_dict(torch.load("models/strategy_net.pt"))
    model.eval()

    print("Enter '0' to quit.\n")
    while True:
        print("  Suits: C=Clubs, D=Diamonds, H=Hearts, S=Spades\n")
        street = input("Street (preflop/flop/turn/river)> ").strip().lower()
        if street=="0": return
        if street not in COMMUNITY_COUNTS:
            print(" ✗ Invalid street\n")
            continue

        needed = COMMUNITY_COUNTS[street]
        while True:
            comm = []
            if needed>0:
                line = input(f"Enter {needed} community cards> ").strip().upper()
                if line=="0": return
                parts = line.split()
                if len(parts)!=needed or not all(is_valid_card(t) for t in parts):
                    print(f" ✗ Need exactly {needed} valid cards\n")
                    continue
                comm = parts
            break

        while True:
            line = input("Your hole cards (2 cards)> ").strip().upper()
            if line=="0": return
            parts = line.split()
            if len(parts)!=2 or not all(is_valid_card(t) for t in parts):
                print(" ✗ Invalid hole cards. Use C7 DA etc.\n")
                continue
            hole = parts
            break

        rs, va = build_round_state_and_actions(street, hole, comm)
        vec = encode_state("me", hole, rs, va)
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw = model(x).numpy().flatten()

        legal = {v["action"] for v in va}
        mask = np.array([raw[i] if a in legal else -1e9
                         for i,a in enumerate(ALL_ACTIONS)], dtype=np.float32)
        if street=="preflop":
            mask[ALL_ACTIONS.index("fold")] = -1e9

        exps = np.exp(mask - mask.max())
        probs = exps / exps.sum()
        idx = int(np.argmax(probs))
        choice = ALL_ACTIONS[idx]

        print_probs(raw, probs, choice)

if __name__=="__main__":
    main()