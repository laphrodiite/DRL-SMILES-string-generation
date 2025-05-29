import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDLogger

from models.generator import SMILESGenerator
from models.predictor import SMILESPredictor
from SMILES_load import TOKENS, token_to_idx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 100

RDLogger.DisableLog('rdApp.*')

# --- Load models ---
gen_pre = SMILESGenerator(len(TOKENS)).to(device)
gen_pre.load_state_dict(torch.load("checkpoints/gen_pre.pt", map_location=device))
gen_pre.eval()

gen_rl = SMILESGenerator(len(TOKENS)).to(device)
gen_rl.load_state_dict(torch.load("checkpoints/gen_rl.pt", map_location=device))
gen_rl.eval()

pred = SMILESPredictor(len(TOKENS)).to(device)
pred.load_state_dict(torch.load("checkpoints/pred.pt", map_location=device))
pred.eval()

# --- Utilities ---
def decode(indices):
    return ''.join(TOKENS[i] for i in indices if TOKENS[i] not in ['<PAD>', '<', '\n'])

def is_valid(smi):
    return Chem.MolFromSmiles(smi) is not None

def tensorize(smi, max_len=100):
    idxs = [token_to_idx.get(c, token_to_idx['<PAD>']) for c in smi]
    if len(idxs) > max_len:
        idxs = idxs[:max_len]
    else:
        idxs += [token_to_idx['<PAD>']] * (max_len - len(idxs))
    return torch.LongTensor([idxs])

# --- Sampling & Evaluation ---
def evaluate_generator(generator, name, n=1000):
    valid_smiles, scores, qed_scores = [], [], []

    for _ in range(n):
        with torch.no_grad():
            idxs = generator.sample(max_len=max_len)
        smi = decode(idxs)
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid_smiles.append(smi)
            try:
                t = tensorize(smi).to(device)
                with torch.no_grad():
                    score = pred(t).item()
                scores.append(score)
                qed_scores.append(QED.qed(mol))
            except:
                continue

    print(f"[{name}] Validity: {len(valid_smiles)}/{n} = {100 * len(valid_smiles)/n:.1f}%")
    print(f"[{name}] Mean predicted property: {np.mean(scores):.3f}")
    print(f"[{name}] Mean QED: {np.mean(qed_scores):.3f}")
    return scores, qed_scores

# --- Run evaluation ---
scores_pre, qeds_pre = evaluate_generator(gen_pre, "Pretrained", n=1000)
scores_rl,  qeds_rl  = evaluate_generator(gen_rl, "RL-Finetuned", n=1000)

# --- Visualization ---
sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(scores_pre, kde=True, color='blue', label='Pretrained', stat='density')
sns.histplot(scores_rl,  kde=True, color='green', label='RL-Finetuned', stat='density')
plt.title("Predicted Property Distribution")
plt.xlabel("Predicted Property")
plt.ylabel("Density")
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(qeds_pre, kde=True, color='blue', label='Pretrained', stat='density')
sns.histplot(qeds_rl,  kde=True, color='green', label='RL-Finetuned', stat='density')
plt.title("QED Distribution")
plt.xlabel("QED")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.savefig("results/comparison.png")
plt.show()
