import random
import pickle
import os

class MemoryBuffer:
    def __init__(self, capacity=250):
        self.capacity = capacity
        self.benign_buffer = []
        self.attack_buffer = []
        
        if os.path.exists('artifacts/initial_buffer.pkl'):
            with open('artifacts/initial_buffer.pkl', 'rb') as f:
                data = pickle.load(f)
                self.benign_buffer = data['benign'][:self.capacity]
                self.attack_buffer = data['attack'][:self.capacity]
                
        if not self.benign_buffer and not self.attack_buffer:
            raise RuntimeError("Memory buffer is empty. You must run train_offline.py first to generate artifacts/initial_buffer.pkl.")

    def insert(self, features: list, label: float):
        packet = (features, label)
        # Insertion logic: replace a random index in the respective buffer[cite: 2]
        if label == 0.0:
            if len(self.benign_buffer) < self.capacity:
                self.benign_buffer.append(packet)
            else:
                idx = random.randint(0, self.capacity - 1)
                self.benign_buffer[idx] = packet
        else:
            if len(self.attack_buffer) < self.capacity:
                self.attack_buffer.append(packet)
            else:
                idx = random.randint(0, self.capacity - 1)
                self.attack_buffer[idx] = packet

    def sample_batch(self, size=31):
        # Randomly sample from combined buffer[cite: 2]
        combined = self.benign_buffer + self.attack_buffer
        if len(combined) < size:
            return random.choices(combined, k=size)
        return random.sample(combined, size)