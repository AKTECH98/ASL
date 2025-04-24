import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class GestureDataset(Dataset):
    def __init__(self, root_dir, label_map=None):
        self.samples = []
        self.labels = []
        self.label_names = []

        # Build or use provided label_map
        if label_map is None:
            self.label_map = {}
            labels = sorted(os.listdir(root_dir))
            for idx, label in enumerate(labels):
                self.label_map[label] = idx
        else:
            self.label_map = label_map

        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path): continue
            if label not in self.label_map:
                continue  # skip unknown labels

            for file in os.listdir(label_path):
                if file.endswith('.npy'):
                    self.samples.append(os.path.join(label_path, file))
                    self.labels.append(self.label_map[label])
                    self.label_names.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        # only hand features the last 42x3 freatures:
        hand_features = np.load(self.samples[idx])[:, -126:]



        x = torch.tensor(hand_features, dtype=torch.float32)

        # x = torch.tensor(np.load(self.samples[idx]), dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in sequences])
    labels = torch.tensor(labels)
    return padded, lengths, labels
