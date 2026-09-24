import os
import random
from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ai.regret_net import RegretNet
from ai.strategy_net import StrategyNet
from memory.regret_memory import RegretMemory
from memory.strategy_memory import StrategyMemory
from engine.runner import CFRRunner
from traverse.traverse import compute_counterfactual_regrets

if __name__ == "__main__":
    input_dim = 113  # state vector length (52 hole + 52 board + 4 street + 5 numeric features)
    regret_net = RegretNet(input_dim)
    strategy_net = StrategyNet(input_dim)
    # Initialize replay memories for regret and strategy samples
    regret_memory = RegretMemory()
    strategy_memory = StrategyMemory()
    # Set up the self-play runner with game parameters
    initial_stack = 1000
    small_blind = 10
    ante = 0
    runner = CFRRunner(regret_net, strategy_net, initial_stack=initial_stack, small_blind=small_blind, ante=ante)
    # Hyperparameters for training
    num_iterations = 1
    episodes_per_iteration = 5
    regret_optimizer = optim.Adam(regret_net.parameters(), lr=0.001)
    strategy_optimizer = optim.Adam(strategy_net.parameters(), lr=0.001)
    # Deep CFR training loop
    for it in trange(num_iterations,desc="Training Iterations"):
        # Self-play to collect episodes
        for ep in trange(episodes_per_iteration,desc=f"Self-Play Iter {it+1}", leave=False):
            players, payoffs = runner.play_episode()
            # Compute regrets for this episode
            regret_samples = compute_counterfactual_regrets(players, payoffs,
                                                           initial_stack=runner.initial_stack,
                                                           small_blind=runner.small_blind,
                                                           ante=runner.ante,
                                                           blind_structure=runner.blind_structure)
            # Store regrets and strategy data from the episode
            for state_vec, regret_vec in regret_samples:
                regret_vec = np.clip(regret_vec, -10, 10)
                regret_memory.add(state_vec, regret_vec)
            for p in players:
                for decision in p.episode_history:
                    strategy_memory.add(decision["state_vec"], decision["strategy"])
        # Train the regret network on collected regret samples
        if len(regret_memory) > 0:
            loss_fn = nn.MSELoss()
            batch_size = 2048
            random.shuffle(regret_memory.memory)
            for i in range(0, len(regret_memory.memory), batch_size):
                batch = regret_memory.memory[i:i+batch_size]
                state_batch = torch.tensor(np.array([s for (s, _) in batch]), dtype=torch.float32)
                target_regret_batch = torch.tensor(np.array([r for (_, r) in batch]), dtype=torch.float32)
                pred_regret = regret_net(state_batch)
                loss = loss_fn(pred_regret, target_regret_batch)
                regret_optimizer.zero_grad()
                loss.backward()
                regret_optimizer.step()
        # Clear regret memory (do not carry over to next iteration)
        regret_memory.clear()
    # After iterations, train the strategy network on all accumulated strategy samples (average strategy)
    if len(strategy_memory) > 0:
        batch_size = 2048
        # for s, pi in strategy_memory.memory[:10]:
        #     print(pi)
        for epoch in range(3):
            random.shuffle(strategy_memory.memory)
            for i in range(0, len(strategy_memory.memory), batch_size):
                batch = strategy_memory.memory[i:i+batch_size]
                state_batch = torch.tensor(np.array([s for (s, _) in batch]), dtype=torch.float32)
                target_strategy_batch = torch.tensor(np.array([dist for (_, dist) in batch]), dtype=torch.float32)
                pred = strategy_net(state_batch)
                loss = nn.MSELoss()(pred, target_strategy_batch)
                strategy_optimizer.zero_grad()
                loss.backward()
                strategy_optimizer.step()
    # Training complete. The strategy_net can now be used for decisions or evaluation.
    os.makedirs("models", exist_ok=True)
    torch.save(strategy_net.state_dict(), "models/strategy_net.pt")
    print("Training complete")

    import matplotlib.pyplot as plt

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    def compute_accuracy(pred_probs, target_probs):
        pred_actions = torch.argmax(pred_probs, dim=1)
        target_actions = torch.argmax(target_probs, dim=1)
        correct = (pred_actions == target_actions).sum().item()
        return correct / len(pred_actions)

    # Add this inside your final strategy_net training block
    if len(strategy_memory) > 0:
        batch_size = 2048
        for epoch in range(100):  # more epochs for better learning curves
            random.shuffle(strategy_memory.memory)
            train_loss_epoch = 0
            train_correct = 0
            total_train = 0

            for i in range(0, int(0.8 * len(strategy_memory.memory)), batch_size):
                batch = strategy_memory.memory[i:i+batch_size]
                state_batch = torch.tensor(np.array([s for (s, _) in batch]), dtype=torch.float32)
                target_strategy_batch = torch.tensor(np.array([dist for (_, dist) in batch]), dtype=torch.float32)
                pred = strategy_net(state_batch)
                loss = nn.MSELoss()(pred, target_strategy_batch)
                strategy_optimizer.zero_grad()
                loss.backward()
                strategy_optimizer.step()

                train_loss_epoch += loss.item() * len(batch)
                train_correct += (torch.argmax(pred, dim=1) == torch.argmax(target_strategy_batch, dim=1)).sum().item()
                total_train += len(batch)

            # Validation on the remaining 20%
            val_loss = 0
            val_correct = 0
            val_total = 0
            for i in range(int(0.8 * len(strategy_memory.memory)), len(strategy_memory.memory), batch_size):
                batch = strategy_memory.memory[i:i+batch_size]
                state_batch = torch.tensor(np.array([s for (s, _) in batch]), dtype=torch.float32)
                target_strategy_batch = torch.tensor(np.array([dist for (_, dist) in batch]), dtype=torch.float32)
                with torch.no_grad():
                    pred = strategy_net(state_batch)
                    val_loss += nn.MSELoss()(pred, target_strategy_batch).item() * len(batch)
                    val_correct += (torch.argmax(pred, dim=1) == torch.argmax(target_strategy_batch, dim=1)).sum().item()
                    val_total += len(batch)

            # Store metrics
            train_loss_history.append(train_loss_epoch / total_train)
            val_loss_history.append(val_loss / val_total)
            train_acc_history.append(train_correct / total_train)
            val_acc_history.append(val_correct / val_total)

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(strategy_net.state_dict(), "models/strategy_net.pt")
    print("Training complete")

    # Plotting training/validation accuracy and loss
    epochs = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc_history, label='train')
    plt.plot(epochs, val_acc_history, label='valid')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("model accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss_history, label='train')
    plt.plot(epochs, val_loss_history, label='valid')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("model loss")
    plt.legend()

    plt.tight_layout()
    plt.show()