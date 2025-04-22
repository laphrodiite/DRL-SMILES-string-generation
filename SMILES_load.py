import pandas as pd
from rdkit import Chem
from tqdm import tqdm  # Progress bar

# Load from local file
local_file = "250k_rndm_zinc_drugs_clean_3.csv"
zinc_df = pd.read_csv(local_file, sep=',', usecols=['smiles']).sample(10000)  # Only load 'smiles' column
zinc_df.to_csv("zinc_sample_10k.csv", index=False)
print(f"Loaded {len(zinc_df)} molecules. Example:\n{zinc_df.head()}")

# Validate and clean-up SMILES
def validate_smiles(smiles_list):
    valid_smiles = []
    for smi in tqdm(smiles_list, desc="Validating SMILES"):
        mol = Chem.MolFromSmiles(smi)
        if mol:  # Only keep valid molecules
            valid_smiles.append(smi)
    return valid_smiles

valid_smiles = validate_smiles(zinc_df['smiles'].tolist())
print(f"Valid SMILES: {len(valid_smiles)}/{len(zinc_df)}")

# Build vocabulary with tokenization
tokens = set()
for smi in valid_smiles:
    tokens.update(list(smi))
vocab = sorted(tokens) + ["<PAD>", "<START>", "<END>"]
token_to_idx = {t:i for i,t in enumerate(vocab)}

print(f"Vocabulary size: {len(vocab)}")
print(f"Example tokens: {list(vocab)[:10]}...")

# Create dataset
import torch
from torch.utils.data import Dataset, DataLoader

class SmilesDataset(Dataset):
    def __init__(self, smiles, token_to_idx, max_len=100):
        self.smiles = smiles
        self.token_to_idx = token_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        # Add start/end tokens and pad
        tokens = ["<START>"] + list(smile) + ["<END>"]
        tokens = tokens[:self.max_len] + ["<PAD>"] * (self.max_len - len(tokens))
        indices = [self.token_to_idx[t] for t in tokens]
        return torch.tensor(indices[:-1]), torch.tensor(indices[1:])  # (input, target)

# Initialize DataLoader
dataset = SmilesDataset(valid_smiles, token_to_idx)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Verify one batch
for x, y in dataloader:
    print(f"Batch shape: {x.shape}, {y.shape}")  # Should be (64, max_len)
    break