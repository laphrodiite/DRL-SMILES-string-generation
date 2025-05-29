import argparse
import torch
from torch import optim
import torch.nn as nn
from models.generator import SMILESGenerator
from models.predictor import SMILESPredictor
from SMILES_load import load_and_process, SMILESDataset, filter_by_tokens, TOKENS, token_to_idx
from torch.utils.data import Dataset, DataLoader, random_split
from SMILES_env import SMILESEnv
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter

class PredictorDataset(Dataset):
    def __init__(self, df, max_len):
        # df must have columns ['smiles', 'property']
        self.smiles = df['smiles'].tolist()
        self.prop   = df['property'].astype(float).tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.prop)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        idxs = [token_to_idx[c] for c in smi]
        # pad/truncate
        if len(idxs) > self.max_len:
            idxs = idxs[:self.max_len]
        else:
            idxs += [token_to_idx['<PAD>']] * (self.max_len - len(idxs))
        return torch.LongTensor(idxs), torch.tensor(self.prop[idx], dtype=torch.float32)


def train_predictor(args):
    # 1. Load & filter
    df = load_and_process(args.prop_csv)
    df = filter_by_tokens(df)
    df = df.rename(columns={args.prop_column: 'property'})

    # 2. Build dataset + train/val split
    full_ds = PredictorDataset(df, max_len=args.max_len)
    val_size = int(len(full_ds) * args.val_frac)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # 3. Model / optimizer / loss
    device = torch.device(args.device)
    model  = SMILESPredictor(len(TOKENS),
                             emb_dim=args.emb_dim,
                             hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        # — train —
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= train_size

        # — validate —
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * x.size(0)
        val_loss /= val_size

        print(f"[Epoch {epoch}] Train MSE: {train_loss:.4f} Val MSE: {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.pred_ckpt)
            print(f"  ↳ Saved new best predictor (val {best_val:.4f})")

    print(f"Done. Best Val MSE: {best_val:.4f}")

def pretrain_generator(args):
    df = load_and_process(args.data_csv, n_samples=args.n_pretrain)
    df = filter_by_tokens(df)
    dataset = SMILESDataset(df, max_len=args.max_len)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    gen = SMILESGenerator(len(TOKENS)).to(args.device)
    opt = optim.Adam(gen.parameters(), lr=args.lr)

    # supervised pretrain
    for epoch in range(args.pre_epochs):
        gen.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(args.device)
            # shift inputs/targets
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            logits, _ = gen(inp)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=token_to_idx['<PAD>']
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: loss {total_loss/len(loader):.4f}")
    torch.save(gen.state_dict(), args.gen_pre_ckpt)


import torch
import torch.nn.functional as F
from models.generator import SMILESGenerator
from SMILES_env import SMILESEnv
from torch.distributions import Categorical
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import QED

def train_with_reinforce(args):
    import torch
    import torch.nn.functional as F
    from torch.distributions import Categorical
    from rdkit import Chem
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    from models.generator import SMILESGenerator
    from SMILES_env import SMILESEnv
    from SMILES_load import TOKENS, token_to_idx

    device = torch.device(args.device)

    # Load pretrained generator
    generator = SMILESGenerator(len(token_to_idx)).to(device)
    generator.load_state_dict(torch.load(args.gen_pre_ckpt))
    generator.train()

    # Simple warm-up reward: reward valid SMILES only
    def reward_fn(smiles, pred=None):
        stripped = smiles.replace('<', '').replace('\n', '').strip()
        if not stripped or len(stripped) < 5:
            return -1.0
        mol = Chem.MolFromSmiles(stripped)
        return 1.0 if mol is not None else -1.0

    env = SMILESEnv(args.pred_ckpt, reward_fn, max_len=args.max_len)
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir="runs/reinforce_smiles")

    baseline = None
    all_rewards = []
    best_reward = -float('inf')

    vocab_size = len(TOKENS)
    mask = torch.full((vocab_size,), float('-inf'), device=device)
    valid_token_ids = torch.tensor(list(token_to_idx.values()), dtype=torch.long).to(device)
    mask[valid_token_ids] = 0

    for step in range(args.rl_steps):
        log_probs, entropies = [], []
        state, _ = env.reset()
        hidden = None

        for t in range(args.max_len):
            current_token = torch.tensor([[state[t]]], device=device)  # [1, 1]
            logits, hidden = generator(current_token, hidden)
            logits = logits[:, -1, :]  # [1, vocab]

            # Apply token mask (e.g., block <PAD>)
            logits += mask

            # Force first token to be one of C, O, N
            if t == 0:
                logits[:] = -1e9
                for tok in ['C', 'O', 'N']:
                    logits[0, token_to_idx[tok]] = 1.0

            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            state, reward, terminated, truncated, _ = env.step(action.item())
            log_probs.append(log_prob)
            entropies.append(entropy)

            if terminated or truncated:
                break

        total_reward = reward
        total_reward = max(min(total_reward, 1.0), -1.0)  # clamp reward for stability

        if baseline is None:
            baseline = total_reward
        else:
            baseline = 0.9 * baseline + 0.1 * total_reward

        advantage = total_reward - baseline
        policy_loss = -torch.stack(log_probs).sum() * advantage
        entropy_bonus = torch.stack(entropies).sum()
        loss = policy_loss - 0.01 * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_rewards.append(total_reward)

        # Logging
        if step % 100 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            print(f"[Step {step}] Avg Reward: {avg_reward:.3f}, Last: {total_reward:.3f}")
            writer.add_scalar("reward/avg_100", avg_reward, step)
            writer.add_scalar("reward/latest", total_reward, step)

            # Optional: SMILES preview
            smi = ''.join(TOKENS[state[i]] for i in range(len(state))
                          if TOKENS[state[i]] not in ['<PAD>', '<', '\n'])
            print(f"Sampled SMILES: {smi}")

        # Save best model so far
        if step % 1000 == 0 and step > 0:
            avg_reward = np.mean(all_rewards[-100:])
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(generator.state_dict(), args.gen_rl_ckpt)
                print(f"  ↳ Saved new best model at step {step} with reward {avg_reward:.3f}")

    writer.close()
    print("✅ Reinforce training complete.")

if __name__ == "__main__":
    # ✏️ Instead of argparse, build one “args” object by hand:
    args = SimpleNamespace(
        # common
        max_len=100,
        device="cuda" if torch.cuda.is_available() else "cpu",

        # 1) generator pretrain
        data_csv="250k_rndm_zinc_drugs_clean_3.csv",
        n_pretrain=10000,
        pre_epochs=10,
        batch=64,
        lr=1e-3,
        gen_pre_ckpt="checkpoints/gen_pre.pt",

        # 2) predictor
        prop_csv="250k_rndm_zinc_drugs_clean_3.csv",
        prop_column="logP",
        pred_ckpt="checkpoints/pred.pt",
        emb_dim=100,
        hidden_dim=100,
        batch_size=128,
        epochs=20,
        val_frac=0.2,

        # 3) RL
        target_prop=2.5,
        rl_steps=10000,
        tb_log=None,
        gen_rl_ckpt="checkpoints/gen_rl.pt",
    )
    from rdkit import RDLogger

# Turn off RDKit warnings and error messages
    RDLogger.DisableLog('rdApp.*')
    # ▶️ Run whatever phase you want, in sequence or pick one:
    #pretrain_generator(args)
    #train_predictor(args)
    train_with_reinforce(args)
