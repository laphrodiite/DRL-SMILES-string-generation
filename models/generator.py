import torch
import torch.nn as nn
from SMILES_load import TOKENS, token_to_idx
import numpy as np

class SMILESGenerator(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        output, hidden = self.gru(emb, hidden)
        logits = self.fc(output)
        return logits, hidden

    def sample(self, max_len=100, start_token='<'):
        self.eval()
        idx = torch.tensor([[token_to_idx[start_token]]], dtype=torch.long)
        hidden = None
        output_seq = [token_to_idx[start_token]]

        for _ in range(max_len):
            logits, hidden = self.forward(idx, hidden)
            probs = torch.softmax(logits[:, -1], dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            token = idx.item()
            output_seq.append(token)
            if TOKENS[token] == '\n':
                break
        return output_seq
