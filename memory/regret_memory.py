import random

class RegretMemory:
    def __init__(self):
        self.memory = []

    def add(self, state_vector, regret_vector):
        """Add a new regret sample to memory."""
        self.memory.append((state_vector, regret_vector))

    def sample(self, batch_size):
        """Randomly sample a batch of regret samples."""
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def clear(self):
        """Clear all stored samples."""
        self.memory = []

    def __len__(self):
        return len(self.memory)