import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Given tokens (with added <PAD> for handling variable lengths)
TOKENS = [
    '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
    '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
    '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n', '<PAD>'
]
token_to_idx = {t: i for i, t in enumerate(TOKENS)}

def load_and_process(csv_path, n_samples=None):
    """Load, validate, and format SMILES data"""
    df = pd.read_csv(csv_path)
    
    # Sample if requested
    if n_samples:
        df = df.sample(n_samples)
    
    # Validate SMILES
    valid_data = []
    for _, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                # Add newline terminator
                formatted_smi = row['smiles'].strip() + '\n'
                valid_data.append({
                    'smiles': formatted_smi,
                    'logP': row['logP'],
                    'qed': row['qed'],
                    'SAS': row['SAS']
                })
        except:
            continue
    
    return pd.DataFrame(valid_data)

def filter_by_tokens(df):
    """Filter SMILES with invalid characters"""
    valid = []
    for smi in df['smiles']:
        if all(c in token_to_idx for c in smi):
            valid.append(smi)
        else:
            invalid_chars = set(smi) - set(token_to_idx.keys())
            print(f"Removed SMILES with invalid chars {invalid_chars}: {smi.strip()}")
    return df[df['smiles'].isin(valid)]

class SMILESDataset(Dataset):
    def __init__(self, df, max_len=None):
        self.df = df
        self.max_len = max_len or self._calculate_max_length()
        
    def _calculate_max_length(self):
        lengths = self.df['smiles'].apply(len)
        return int(np.percentile(lengths, 95))  # 95th percentile length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        smi = self.df.iloc[idx]['smiles']
        indices = [token_to_idx[c] for c in smi]
        
        # Truncate or pad
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            padding = [token_to_idx['<PAD>']] * (self.max_len - len(indices))
            indices += padding
            
        return torch.LongTensor(indices)

# Usage example
if __name__ == "__main__":
    # Load and process data
    df = load_and_process("250k_rndm_zinc_drugs_clean_3.csv", n_samples=10000)
    filtered_df = filter_by_tokens(df)
    
    # Create dataset and dataloader
    dataset = SMILESDataset(filtered_df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Test output
    sample = next(iter(dataloader))
    print(f"Batch shape: {sample.shape}")
    print("Sample sequence:", [TOKENS[i] for i in sample[0].tolist()[:10]])