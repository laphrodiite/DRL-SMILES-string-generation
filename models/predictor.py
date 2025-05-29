import torch.nn as nn
from SMILES_load import token_to_idx

class SMILESPredictor(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=token_to_idx['<PAD>'])
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)  # Predict logP, QED, etc.

    def forward(self, x):
        emb = self.embedding(x)
        _, (hn, _) = self.lstm(emb)
        x = self.relu(self.fc1(hn[-1]))
        return self.fc2(x).squeeze(-1)  # [B]
