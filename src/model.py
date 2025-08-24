import ast
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

MIN_DIFFICULTY = 10
MAX_DIFFICULTY = 31
MIN_ANGLE = 0
MAX_ANGLE = 70

def parse_tokens(x):
    return x if isinstance(x, (list, tuple)) else ast.literal_eval(x)

def sample_logits(logits, temperature):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

class Model:

    def __init__(self, holds_csv="holds.csv", roles_csv="hold_roles.csv"):
        holds_df = pd.read_csv(holds_csv)
        roles_df = pd.read_csv(roles_csv)

        self.holds = torch.tensor(holds_df["index"].values)
        self.roles = torch.tensor(roles_df["index"].values) # 0: (s)tart, 1: (m)iddle, 2: (e)nd, 3: (f)oot, 4: (.)special
        self.num_tokens = 1 + len(self.holds) * len(self.roles) # +1 for BOS/EOS=0

    def train(self, climbs_csv="climbs_train.csv", **kwargs):
        raise NotImplementedError("train() method must be implemented by subclasses")

    def generate(self, difficulty, angle, max_length=None, **kwargs):
        """Generate a sequence of holds and roles"""
        raise NotImplementedError("generate() method must be implemented by subclasses")

class ARModel(Model):

    def train(self, climbs_csv="climbs_train.csv", **kwargs):
        raise NotImplementedError("train() method must be implemented by subclasses")

    def generate(self, difficulty, angle, temperature, max_length=40, **kwargs):
        """Generate a sequence of tokens autoregressively.
        max_length: optional cap on the number of generated tokens (excluding BOS/EOS).
        """
        tokens = [0]  # BOS/EOS token
        while True:
            # Stop if we've generated max_length tokens (excluding BOS)
            if max_length is not None and (len(tokens) - 1) >= max_length:
                break
            logits = self.next_token_logits(difficulty, angle, tokens, **kwargs)
            next_token = sample_logits(logits, temperature=temperature).item()
            tokens.append(next_token)
            if next_token == 0:
                break
        return tokens

    def next_token_logits(self, difficulty, angle, prev_tokens, **kwargs):
        raise NotImplementedError("next_token_logits() method must be implemented by subclasses")

    @torch.no_grad()
    def nll(self, model, df):
        """Compute negative log-likelihood (nats/token) on a dataframe of climbs with 'tokens', 'difficulty', and 'angle' columns"""
        total_loss = 0.0
        total_tokens = 0
        for _, row in df.iterrows():
            difficulty = row["difficulty"]
            angle = row["angle"]
            tokens = parse_tokens(row["tokens"])
            for i in range(len(tokens) - 1):
                logits = model.next_token_logits(difficulty, angle, tokens[:i+1])
                target = torch.tensor([tokens[i+1]])
                total_loss += F.cross_entropy(logits.unsqueeze(0), target, reduction='sum').item()
                total_tokens += 1
        nll = total_loss / max(1, total_tokens)      # nats/token
        return nll

class BigramModel(ARModel):

    def __init__(self, holds_csv="holds.csv", roles_csv="hold_roles.csv", num_difficulty_bins=1+(MAX_DIFFICULTY-MIN_DIFFICULTY)//2, num_angle_bins=1+(MAX_ANGLE-MIN_ANGLE)//5):
        super().__init__(holds_csv, roles_csv)
        self.difficulty_bins = torch.linspace(MIN_DIFFICULTY, MAX_DIFFICULTY, num_difficulty_bins)
        self.angle_bins = torch.linspace(MIN_ANGLE, MAX_ANGLE, num_angle_bins)
        self.counts = torch.ones((num_difficulty_bins, num_angle_bins, self.num_tokens, self.num_tokens), dtype=torch.int) # add-one smoothing

    def train(self, climbs_csv="climbs_train.csv", **kwargs):
        climbs_df = pd.read_csv(climbs_csv)
        self.counts.fill_(1) # reset counts with add-one smoothing
        for _, row in climbs_df.iterrows():
            difficulty = row["difficulty"]
            angle = row["angle"]
            tokens = parse_tokens(row["tokens"])
            d_idx = self._bin_idx(difficulty, self.difficulty_bins)
            a_idx = self._bin_idx(angle, self.angle_bins)
            for i in range(len(tokens) - 1):
                self.counts[d_idx, a_idx, tokens[i], tokens[i+1]] += 1
    
    def next_token_logits(self, difficulty, angle, prev_tokens, **kwargs):
        d_idx = self._bin_idx(difficulty, self.difficulty_bins)
        a_idx = self._bin_idx(angle, self.angle_bins)
        last_token = prev_tokens[-1]
        logits = self.counts[d_idx, a_idx, last_token].float().log()
        return logits
    
    def _bin_idx(self, x, bins):
        i = torch.bucketize(torch.tensor([x], dtype=bins.dtype), bins).item() - 1
        return max(0, min(i, len(bins) - 1))
