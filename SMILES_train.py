# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from types import SimpleNamespace
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import QED, Crippen
from rdkit import RDLogger

# Local imports
from models.generator import SMILESGenerator
from models.predictor import SMILESPredictor
from SMILES_env import SMILESEnv
from SMILES_load import (
    load_and_process, SMILESDataset,
    filter_by_tokens, TOKENS, token_to_idx
)

# --------------------- Dataset Class for Property Prediction ---------------------
class PredictorDataset(Dataset):
    """
    Custom PyTorch Dataset for SMILES + property pairs.
    Each sample is a tokenized SMILES sequence and its associated property value (e.g., logP).
    """
    def __init__(self, df, max_len):
        self.smiles = df['smiles'].tolist()
        self.prop = df['property'].astype(float).tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.prop)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        idxs = [token_to_idx[c] for c in smi]
        # Pad the sequence to max_len (right-padding with <PAD> token)
        idxs = idxs[:self.max_len] + [token_to_idx['<PAD>']] * max(0, self.max_len - len(idxs))
        return torch.LongTensor(idxs), torch.tensor(self.prop[idx], dtype=torch.float32)

# --------------------- Supervised Pretraining of the Generator ---------------------
def pretrain_generator(args):
    """
    Trains the SMILES generator in a supervised fashion to mimic real SMILES sequences.
    Objective: teach the generator to output chemically valid-looking SMILES syntax.
    """
    # Load and clean the SMILES dataset
    df = load_and_process(args.data_csv, n_samples=args.n_pretrain)
    df = filter_by_tokens(df)

    # Create DataLoader for mini-batch training
    dataset = SMILESDataset(df, max_len=args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    # Initialize generator model and optimizer
    gen = SMILESGenerator(len(TOKENS)).to(args.device)
    opt = optim.Adam(gen.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.pre_epochs):
        gen.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(args.device)
            inp, tgt = batch[:, :-1], batch[:, 1:]  # Input & target for next-token prediction
            logits, _ = gen(inp)

            # Compute cross-entropy loss (ignore <PAD>)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=token_to_idx['<PAD>']
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[Pretrain Epoch {epoch}] Loss: {total_loss/len(loader):.4f}")

    # Save pretrained weights
    torch.save(gen.state_dict(), args.gen_pre_ckpt)

# --------------------- Train the Property Predictor ---------------------
def train_predictor(args):
    """
    Trains a regression model that predicts molecular properties (e.g., logP) from SMILES.
    This model is later used as a reward function in reinforcement learning.
    """
    # Load and rename the target column to "property"
    df = load_and_process(args.prop_csv)
    df = filter_by_tokens(df).rename(columns={args.prop_column: 'property'})

    # Prepare train/validation splits
    full_ds = PredictorDataset(df, max_len=args.max_len)
    val_size = int(len(full_ds) * args.val_frac)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # DataLoaders for batching
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Initialize model and optimizer
    model = SMILESPredictor(len(TOKENS), args.emb_dim, args.hidden_dim).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= train_size

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                val_loss += criterion(model(x), y).item() * x.size(0)
        val_loss /= val_size

        print(f"[Epoch {epoch}] Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.pred_ckpt)
            print(f"  ↳ Saved new best predictor (val {best_val:.4f})")

    print(f"✅ Predictor training done. Best Val MSE: {best_val:.4f}")

# --------------------- Reinforcement Learning Training ---------------------
def train_with_reinforce(args):
    """
    Uses REINFORCE to fine-tune the generator to produce SMILES with desired properties.
    Reward is based on logP and QED of generated molecules.
    """
    # Load pretrained generator
    generator = SMILESGenerator(len(token_to_idx)).to(args.device)
    generator.load_state_dict(torch.load(args.gen_pre_ckpt))
    generator.train()

    # Define custom reward function using RDKit
    def reward_fn(smiles, pred=None):
        smi = smiles.replace('<', '').replace('\n', '').strip()
        if not smi or len(smi) < 5:
            return -1.0
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return -2.0
        logp = Crippen.MolLogP(mol)
        qed = QED.qed(mol)
        return 0.5 * (logp / 5.0) + 0.5 * qed


    env = SMILESEnv(args.pred_ckpt, reward_fn, max_len=args.max_len)
    optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir="runs/reinforce_smiles")

    baseline = None
    all_rewards = []
    best_reward = -float('inf')
    vocab_size = len(TOKENS)

    # Precompute logit mask (valid tokens only)
    mask = torch.full((vocab_size,), float('-inf'), device=args.device)
    mask[torch.tensor(list(token_to_idx.values()), dtype=torch.long)] = 0

    # Training loop
    for step in range(args.rl_steps):
        log_probs, entropies = [], []
        state, _ = env.reset()
        hidden = None

        # Rollout episode
        for t in range(args.max_len):
            current_token = torch.tensor([[state[t]]], device=args.device)
            logits, hidden = generator(current_token, hidden)
            logits = logits[:, -1, :] + mask

            # Constrain first token
            if t == 0:
                logits[:] = -1e9
                for tok in ['C', 'c', 'O', 'N']:
                    logits[0, token_to_idx[tok]] = 1.0

            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            state, reward, terminated, truncated, _ = env.step(action.item())
            if terminated or truncated:
                break

        # Get total reward and compute REINFORCE loss
        total_reward = reward #max(min(reward, 1.0), -1.0)
        baseline = total_reward if baseline is None else 0.9 * baseline + 0.1 * total_reward
        advantage = total_reward - baseline

        policy_loss = -torch.stack(log_probs).sum() * advantage
        entropy_bonus = torch.stack(entropies).sum()
        loss = policy_loss - 0.05 * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_rewards.append(total_reward)

        # Logging to console and TensorBoard
        if step % 100 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            print(f"[Step {step}] Avg Reward: {avg_reward:.3f}, Last: {total_reward:.3f}")
            writer.add_scalar("reward/avg_100", avg_reward, step)
            writer.add_scalar("reward/latest", total_reward, step)

            smi = ''.join(TOKENS[state[i]] for i in range(len(state))
                          if TOKENS[state[i]] not in ['<PAD>', '<', '\n'])
            print(f"Sampled SMILES: {smi}")

        # Save best-performing generator
        if step % 100 == 0 and step > 0:
            avg_reward = np.mean(all_rewards[-100:])
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(generator.state_dict(), args.gen_rl_ckpt)
                print(f"  ↳ Saved new best model at step {step} with reward {avg_reward:.3f}")

    writer.close()
    print("✅ REINFORCE training complete.")

# --------------------- Main Entry Point ---------------------
if __name__ == "__main__":
    # Silence RDKit warnings
    RDLogger.DisableLog('rdApp.*')

    # Define parameters for all phases
    args = SimpleNamespace(
        max_len=100,
        device="cuda" if torch.cuda.is_available() else "cpu",

        # Generator pretraining
        data_csv="data/250k_rndm_zinc_drugs_clean_3.csv",
        n_pretrain=10000,
        pre_epochs=10,
        batch=64,
        lr=1e-3,
        gen_pre_ckpt="checkpoints/gen_pre.pt",

        # Property predictor training
        prop_csv="data/250k_rndm_zinc_drugs_clean_3.csv",
        prop_column="logP",
        pred_ckpt="checkpoints/pred.pt",
        emb_dim=100,
        hidden_dim=100,
        batch_size=128,
        epochs=20,
        val_frac=0.2,

        # Reinforcement learning
        target_prop=2.5,
        rl_steps=2000,
        tb_log=None,
        gen_rl_ckpt="checkpoints/gen_rl.pt",
    )

    # Run full pipeline
    pretrain_generator(args)
    train_predictor(args)
    train_with_reinforce(args)