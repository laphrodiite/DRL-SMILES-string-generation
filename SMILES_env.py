import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from models.generator import SMILESGenerator
from models.predictor import SMILESPredictor
from SMILES_load import TOKENS, token_to_idx


class SMILESEnv(gym.Env):
    def __init__(self, predictor_ckpt, reward_fn, max_len=100):
        super().__init__()
        self.max_len   = max_len
        self.reward_fn = reward_fn

        # Load predictor model
        self.predictor = SMILESPredictor(len(TOKENS))
        self.predictor.load_state_dict(torch.load(predictor_ckpt))
        self.predictor.eval()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(TOKENS))
        self.observation_space = spaces.Box(
            low=0,
            high=len(TOKENS) - 1,
            shape=(max_len,),
            dtype=np.int64
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.smiles = ['<']
        self.done = False
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        tok = TOKENS[action]
        self.smiles.append(tok)

        # Define done conditions
        terminated = tok == '\n'
        truncated = len(self.smiles) >= self.max_len
        self.done = terminated or truncated

        reward = 0.0
        if self.done:
            seq = ''.join(self.smiles)
            # Check for invalid characters
            try:
                idxs = [token_to_idx[c] for c in seq]
            except KeyError:
                return self._get_obs(), -1.0, terminated, truncated, {}  # softer penalty

            from rdkit import Chem
            mol = Chem.MolFromSmiles(seq.replace('<', '').replace('\n', ''))  # clean up

            # Predict activity (optional predictor model)
            padded = idxs + [token_to_idx['<PAD>']] * (self.max_len - len(idxs))
            t = torch.LongTensor([padded])
            with torch.no_grad():
                pred = self.predictor(t).item()

            # Compute reward
            reward = self.reward_fn(seq, pred) if callable(self.reward_fn) else pred

            # Normalize reward
            reward = reward

            # Bonus for early stop
            if tok == '\n':
                reward += 0.2  


        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info


    def _get_obs(self):
        idxs = [token_to_idx[c] for c in self.smiles]
        padded = idxs + [token_to_idx['<PAD>']] * (self.max_len - len(idxs))
        return np.array(padded, dtype=np.int64)
