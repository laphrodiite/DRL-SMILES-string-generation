import gymnasium as gym
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from SMILES_load import token_to_idx, TOKENS  # Import from your loader

class SMILESEnv(gym.Env):
    def __init__(self, token_to_idx, max_len=80):
        """
        Args:
            token_to_idx: Dictionary mapping tokens to indices (from SMILES_load.py)
            max_len: Maximum allowed SMILES length
        """
        self.token_to_idx = token_to_idx
        self.idx_to_token = {v: k for k, v in token_to_idx.items()}
        self.max_len = max_len
        
        # Action space: Choose any token
        self.action_space = gym.spaces.Discrete(len(token_to_idx))
        
        # Observation space: Current sequence (padded to max_len)
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=len(token_to_idx)-1, 
            shape=(max_len,), 
            dtype=np.int32
        )
        
        # Initialize state tracking
        self.reset()

    def reset(self):
        """Start new episode with empty sequence"""
        self.current_smiles = []
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        """
        Args:
            action: Token index to append
        Returns:
            obs: Updated sequence
            reward: Calculated reward
            done: Termination flag
            info: Additional metrics
        """
        token = self.idx_to_token[action]
        self.current_smiles.append(token)
        self.current_step += 1
        
        # Termination conditions
        done = (token == '\n') or (self.current_step >= self.max_len)
        
        # Reward calculation
        reward = 0
        validity = 0
        qed = 0
        
        if done:
            smile_str = ''.join(self.current_smiles[:-1])  # Exclude \n
            mol = Chem.MolFromSmiles(smile_str)
            
            if mol:
                validity = 1.0
                qed = QED.qed(mol)
                reward = validity + 0.5 * qed
            else:
                reward = -1.0
        
        info = {
            'valid': bool(validity),
            'qed': qed,
            'smiles': ''.join(self.current_smiles[:-1]) if done else None
        }
        
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """Convert current state to fixed-length observation"""
        obs = [self.token_to_idx[t] for t in self.current_smiles]
        obs += [self.token_to_idx['<PAD>']] * (self.max_len - len(obs))
        return np.array(obs, dtype=np.int32)

    def render(self, mode='human'):
        """Optional: Visualize current SMILES"""
        print(''.join(self.current_smiles))

    def close(self):
        """Cleanup resources if needed"""
        pass