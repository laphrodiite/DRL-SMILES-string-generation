import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from rdkit.Chem import Draw
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
def save_molecule_images(smiles_list, name, max_mols=20):
    """Save images of generated molecules from SMILES strings."""
    os.makedirs(f"results/{name}_molecules", exist_ok=True)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list[:max_mols] if Chem.MolFromSmiles(smi) is not None]

    for i, mol in enumerate(mols):
        img = Draw.MolToImage(mol, size=(300, 300))
        img.save(f"results/{name}_molecules/mol_{i+1}.png")

    # Optional: one grid image for preview
    grid = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200), legends=[f"{i+1}" for i in range(len(mols))])
    grid.save(f"results/{name}_molecules/grid.png")
    print(f"[{name}] Molecule images saved to 'results/{name}_molecules/'")

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

# --- Evaluation ---
def evaluate_generator(generator, name, n=1000):
    valid_smiles, scores, qed_scores, token_freqs = [], [], [], []

    for _ in range(n):
        with torch.no_grad():
            idxs = generator.sample(max_len=max_len)
        smi = decode(idxs)
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid_smiles.append(smi)
            token_freqs.extend(list(smi))
            try:
                t = tensorize(smi).to(device)
                with torch.no_grad():
                    score = pred(t).item()
                scores.append(score)
                qed_scores.append(QED.qed(mol))
            except:
                continue

    # Save SMILES examples
    with open(f"results/{name}_generated_smiles.txt", "w") as f:
        for smi in valid_smiles:
            f.write(smi + "\n")

    print(f"\n[{name}] Validity: {len(valid_smiles)}/{n} = {100 * len(valid_smiles)/n:.1f}%")
    print(f"[{name}] Mean predicted property: {np.mean(scores):.3f}")
    print(f"[{name}] Mean QED: {np.mean(qed_scores):.3f}")
    print(f"[{name}] Most frequent tokens: {Counter(token_freqs).most_common(10)}")

    # Print a few example SMILES strings
    print(f"\n[{name}] Sample generated SMILES:")
    for smi in valid_smiles[:10]:
        print(" ", smi)

    # Save molecule images
    save_molecule_images(valid_smiles, name)

    return scores, qed_scores, Counter(token_freqs)



# --- Run evaluation ---
scores_pre, qeds_pre, freq_pre = evaluate_generator(gen_pre, "Pretrained", n=1000)
scores_rl,  qeds_rl,  freq_rl  = evaluate_generator(gen_rl, "RL-Finetuned", n=1000)

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

# --- Plot Token Frequencies ---
def plot_token_frequencies(counter, title, name):
    most_common = counter.most_common(20)
    tokens, counts = zip(*most_common)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=list(tokens), y=list(counts), palette="mako")
    plt.title(f"{title} (Top 20 Tokens)")
    plt.xlabel("Token")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"results/{name}_token_distribution.png")
    plt.show()

plot_token_frequencies(freq_pre, "Pretrained Token Frequency", "pretrained")
plot_token_frequencies(freq_rl, "RL-Finetuned Token Frequency", "rl")
