import numpy as np
import pandas as pd
import random
import ast

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Generator:
    def __init__(self):
        self.model = None
        self.all_hold_ids = None
        self.all_hold_xys = None
        self.role_id_to_letter = None
        self.role_letter_to_index = None  # maps 's','m','e','f' -> {0..3}
        self.role_index_to_letter = None  # list index->{'s','m','e','f'}
        self._load_hold_data()

    def train(self, df: pd.DataFrame):
        raise NotImplementedError

    def generate(self, length: int, temp: float = 1.0, angle: int = None, difficulty: str = None):
        raise NotImplementedError

    def _pick(self, weights, temp=1.0):
        w = np.asarray(weights, dtype=np.float64)
        if temp != 1.0:
            w = np.power(w, 1.0 / temp)
        
        if np.sum(w) == 0: # Guard against all-zero weights
            return random.randrange(len(w))
            
        return random.choices(range(len(w)), weights=w, k=1)[0]

    def _sample_from_logits(self, logits: torch.Tensor, temp: float = 1.0) -> int:
        """Sample an index from unnormalized logits with optional temperature.

        - logits: 1D tensor of shape (V,)
        - temp <= 0 -> argmax
        - temp > 0 -> softmax(logits/temp) then multinomial
        """
        if logits.ndim != 1:
            logits = logits.reshape(-1)
        if temp <= 0:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits / float(temp), dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    # ---- Shared small utilities for simple Markov-like generators ----
    def _logits_from_start_counts(self, start_counts: np.ndarray) -> torch.Tensor:
        s = float(np.clip(start_counts.sum(), 1e-8, None))
        probs = (start_counts / s).astype(np.float32)
        return torch.log(torch.from_numpy(probs))

    def _logits_from_row_counts(self, counts: np.ndarray) -> torch.Tensor:
        row_sums = counts.sum(axis=1, keepdims=True)
        probs = counts / np.clip(row_sums, 1e-8, None)
        return torch.log(torch.tensor(probs, dtype=torch.float32))

    def _monitor_ce(self, logits_start: torch.Tensor, logits_trans: torch.Tensor,
                    start_targets: list[int], trans_srcs: list[int], trans_tgts: list[int],
                    verbose: bool, tag: str):
        if not verbose:
            return
        tokens = 0
        nll = 0.0
        if start_targets:
            t = torch.tensor(start_targets, dtype=torch.long)
            ls = logits_start.unsqueeze(0).expand(len(start_targets), -1)
            nll += float(F.cross_entropy(ls, t, reduction="sum").item())
            tokens += len(start_targets)
        if trans_srcs:
            src = torch.tensor(trans_srcs, dtype=torch.long)
            tgt = torch.tensor(trans_tgts, dtype=torch.long)
            sel = logits_trans.index_select(0, src)
            nll += float(F.cross_entropy(sel, tgt, reduction="sum").item())
            tokens += len(trans_srcs)
        if tokens:
            print(f"[{tag}] loss/token: {nll/tokens:.4f} (tokens: {tokens})")

    def _generate_markov(self, logits_start: torch.Tensor, logits_trans: torch.Tensor,
                         length: int, temp: float) -> list[int]:
        first = self._sample_from_logits(logits_start, temp=temp)
        out = [first]
        for _ in range(length - 1):
            out.append(self._sample_from_logits(logits_trans[out[-1]], temp=temp))
        return out

    def _parse_holds_indices(self, holds_indices_str):
        return ast.literal_eval(holds_indices_str)

    def _parse_roles_from_holds_xy(self, holds_xy_str):
        # holds_xy is a stringified list of triples (x, y, role_letter)
        return [t[2] for t in ast.literal_eval(holds_xy_str)]

    def _load_hold_data(self):
        holds_df = pd.read_csv("holds.csv")
        self.all_hold_ids = list(holds_df["hold_id"])
        self.all_hold_xys = list(holds_df[["x", "y"]].itertuples(index=False, name=None))
        
        roles_df = pd.read_csv("hold_roles.csv")
        self.role_id_to_letter = dict(zip(roles_df["id"], roles_df["letter"]))
        letters = list(roles_df["letter"])
        self.role_index_to_letter = letters
        self.role_letter_to_index = {l: i for i, l in enumerate(letters)}

    def hold_id_to_index(self, hold_id: int) -> int:
        return self.all_hold_ids.index(hold_id)

    def index_to_hold_id(self, idx: int) -> int:
        return self.all_hold_ids[idx]

    def index_to_xy(self, idx):
        return self.all_hold_xys[idx]

class CooccurrenceGenerator(Generator):
    def train(self, df: pd.DataFrame, alpha: float = 1.0, verbose: bool = True):
        n = len(self.all_hold_ids)
        co_counts = np.full((n, n), alpha, dtype=np.float64)
        start_counts = np.full(n, alpha, dtype=np.float64)
        start_targets, trans_srcs, trans_tgts = [], [], []

        for holds_str in df["holds_indices"]:
            idxs = self._parse_holds_indices(holds_str)
            if not idxs:
                continue
            start_counts[idxs[0]] += 1
            start_targets.append(idxs[0])
            uniq = list(dict.fromkeys(idxs))
            for i in uniq:
                for j in uniq:
                    if j != i:
                        co_counts[i, j] += 1
            for a, b in zip(idxs, idxs[1:]):
                trans_srcs.append(a)
                trans_tgts.append(b)

        logits_trans = self._logits_from_row_counts(co_counts)
        logits_start = self._logits_from_start_counts(start_counts)
        self.model = {"logits_trans": logits_trans, "logits_start": logits_start}
        self._monitor_ce(logits_start, logits_trans, start_targets, trans_srcs, trans_tgts, verbose, tag="CoOcc")

    def generate(self, length: int, temp: float = 1.0, angle: int = None, difficulty: str = None):
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        return self._generate_markov(self.model["logits_start"], self.model["logits_trans"], length, temp)

class BigramGenerator(Generator):
    def train(self, df: pd.DataFrame, alpha: float = 1.0, verbose: bool = True):
        n = len(self.all_hold_ids)
        counts = np.full((n, n), alpha, dtype=np.float64)
        start_counts = np.full(n, alpha, dtype=np.float64)
        start_targets, trans_srcs, trans_tgts = [], [], []

        for _, row in df.iterrows():
            seq = self._parse_holds_indices(row["holds_indices"])
            if not seq:
                continue
            start_counts[seq[0]] += 1
            start_targets.append(seq[0])
            for a, b in zip(seq, seq[1:]):
                counts[a, b] += 1
                trans_srcs.append(a)
                trans_tgts.append(b)

        logits_trans = self._logits_from_row_counts(counts)
        logits_start = self._logits_from_start_counts(start_counts)
        self.model = {"logits_trans": logits_trans, "logits_start": logits_start}
        self._monitor_ce(logits_start, logits_trans, start_targets, trans_srcs, trans_tgts, verbose, tag="Bigram")

    def generate(self, length: int, temp: float = 1.0, angle: int = None, difficulty: str = None):
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        return self._generate_markov(self.model["logits_start"], self.model["logits_trans"], length, temp)


class AutoRegressiveGenerator(Generator):
    """
    A small autoregressive language model over (hold_index, role) pairs.
    - Tokens are pairs (hold_idx in [0..N-1], role in {'s','m','e','f'}).
    - Uses a GRU over sum of hold and role embeddings (+ positional embeddings).
    - Trained with cross-entropy to predict the next token at each position.
    - Generation: first role is forced to 's'; stops when role 'e' is sampled or
      when max_length is reached. Temperature controls sampling.
    Context length is capped at context_len (default 40).
    """

    class _ARM(nn.Module):
        def __init__(self, n_holds: int, n_roles: int, d_model: int = 128, hidden_size: int = 256,
                     num_layers: int = 1, context_len: int = 40):
            super().__init__()
            self.n_holds = n_holds
            self.n_roles = n_roles
            self.d_model = d_model
            self.context_len = context_len

            self.hold_emb = nn.Embedding(n_holds, d_model)
            self.role_emb = nn.Embedding(n_roles, d_model)
            self.pos_emb = nn.Embedding(context_len + 1, d_model)  # +1 for BOS position
            self.bos = nn.Parameter(torch.zeros(d_model))

            self.gru = nn.GRU(input_size=d_model, hidden_size=hidden_size, num_layers=num_layers,
                               batch_first=True)
            self.head_hold = nn.Linear(hidden_size, n_holds)
            self.head_role = nn.Linear(hidden_size, n_roles)

        def _embed_tokens(self, hold_seq, role_seq):
            # hold_seq/role_seq: (B,T) possibly T=0
            B, T = hold_seq.shape
            if T == 0:
                # Only BOS
                x = self.bos.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B,1,D)
                return x
            h = self.hold_emb(hold_seq)
            r = self.role_emb(role_seq)
            x = h + r  # (B,T,D)
            # positions: 1..T for tokens (0 is reserved for BOS)
            pos_idx = torch.arange(1, T + 1, device=hold_seq.device).unsqueeze(0).expand(B, T)
            x = x + self.pos_emb(pos_idx)
            # prepend BOS at position 0
            bos = self.bos.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            x = torch.cat([bos, x], dim=1)  # (B,T+1,D)
            return x

        def forward(self, hold_seq, role_seq):
            # Training forward: returns logits for targets at positions 0..T-1 (T positions)
            # Inputs are the T tokens; BOS is prepended internally.
            B, T = hold_seq.shape
            x = self._embed_tokens(hold_seq, role_seq)  # (B,T+1,D) or (B,1,D) if T=0
            y, _ = self.gru(x)  # (B,T+1,H)
            if T == 0:
                out = y  # (B,1,H)
                logits_hold = self.head_hold(out)
                logits_role = self.head_role(out)
                return logits_hold, logits_role
            # drop the final step to align with targets 0..T-1
            out = y[:, :-1, :]  # (B,T,H)
            logits_hold = self.head_hold(out)
            logits_role = self.head_role(out)
            return logits_hold, logits_role

        def next_logits(self, hold_seq, role_seq):
            # Returns logits for the next token (position T) given current sequence of length T
            B, T = hold_seq.shape
            x = self._embed_tokens(hold_seq, role_seq)  # (B,T+1,D)
            y, _ = self.gru(x)                          # (B,T+1,H)
            last = y[:, -1, :]                          # (B,H)
            logits_hold = self.head_hold(last)
            logits_role = self.head_role(last)
            return logits_hold, logits_role

    def __init__(self, context_len=40, d_model=128, hidden_size=256,
                 num_layers=1, device=None):
        super().__init__()
        if torch is None:
            raise ImportError("PyTorch is required for AutoRegressiveGenerator. Please install torch.")
        self.context_len = context_len
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        # Will initialize model in train() after hold/role sizes are known
        self.model = None

    def _encode_roles(self, letters):
        return [self.role_letter_to_index[l] for l in letters if l in self.role_letter_to_index]

    def _build_samples(self, df: pd.DataFrame):
        samples = []
        for _, row in df.iterrows():
            holds_idx = self._parse_holds_indices(row["holds_indices"]) if isinstance(row["holds_indices"], str) else row["holds_indices"]
            roles_letters = self._parse_roles_from_holds_xy(row["holds_xy"]) if isinstance(row["holds_xy"], str) else [t[2] for t in row["holds_xy"]]
            if not holds_idx or not roles_letters:
                continue
            # Align lengths just in case
            L = min(len(holds_idx), len(roles_letters))
            holds_idx = holds_idx[:L]
            role_ids = self._encode_roles(roles_letters[:L])
            if len(holds_idx) < 1 or len(holds_idx) != len(role_ids):
                continue
            samples.append((holds_idx, role_ids))
        return samples

    def train(self, df: pd.DataFrame, epochs=3, lr=3e-3, weight_decay=1e-2,
              max_samples=None, verbose: bool = True):
        n_holds = len(self.all_hold_ids)
        n_roles = len(self.role_index_to_letter)
        self.model = AutoRegressiveGenerator._ARM(
            n_holds=n_holds,
            n_roles=n_roles,
            d_model=self.d_model,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            context_len=self.context_len,
        ).to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_hold_fn = nn.CrossEntropyLoss()
        loss_role_fn = nn.CrossEntropyLoss()

        samples = self._build_samples(df)
        if max_samples is not None:
            samples = samples[:max_samples]

        self.model.train()
        for ep in range(epochs):
            random.shuffle(samples)
            total_loss = 0.0
            token_count = 0
            for holds_idx, role_ids in samples:
                # Clip context to the last context_len tokens for long sequences
                holds_tensor = torch.tensor(holds_idx[-self.context_len:], dtype=torch.long, device=self.device).unsqueeze(0)
                roles_tensor = torch.tensor(role_ids[-self.context_len:], dtype=torch.long, device=self.device).unsqueeze(0)
                logits_hold, logits_role = self.model.forward(holds_tensor, roles_tensor)  # (1,T,*) predicts current tokens with BOS
                T = logits_hold.shape[1]
                target_h = holds_tensor[:, :T].reshape(-1)
                target_r = roles_tensor[:, :T].reshape(-1)
                loss_h = loss_hold_fn(logits_hold.reshape(-1, logits_hold.shape[-1]), target_h)
                loss_r = loss_role_fn(logits_role.reshape(-1, logits_role.shape[-1]), target_r)
                loss = loss_h + loss_r

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * T
                token_count += T

            if verbose and token_count > 0:
                print(f"[AR] epoch {ep+1}/{epochs} - loss/token: {total_loss/token_count:.4f} (tokens: {token_count})")

    # Sampling helper is inherited from Generator

    def generate(self, max_length: int = 40, temp: float = 1.0, angle: int = None, difficulty: str = None):
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        self.model.eval()

        holds_ctx = torch.empty((1, 0), dtype=torch.long, device=self.device)
        roles_ctx = torch.empty((1, 0), dtype=torch.long, device=self.device)

        result: list[tuple[int, str]] = []

        # First token: force role='s'
        with torch.no_grad():
            logits_hold, logits_role = self.model.next_logits(holds_ctx, roles_ctx)
            hold_idx = self._sample_from_logits(logits_hold[0], temp=temp)
            role_idx = self.role_letter_to_index.get('s', 0)
            result.append((hold_idx, 's'))
            holds_ctx = torch.tensor([[hold_idx]], dtype=torch.long, device=self.device)
            roles_ctx = torch.tensor([[role_idx]], dtype=torch.long, device=self.device)

        # Subsequent tokens
        for _ in range(max_length - 1):
            # Keep only last context_len tokens
            if holds_ctx.shape[1] > self.context_len:
                holds_ctx = holds_ctx[:, -self.context_len:]
                roles_ctx = roles_ctx[:, -self.context_len:]
            with torch.no_grad():
                logits_hold, logits_role = self.model.next_logits(holds_ctx, roles_ctx)
                # Sample role first; if 'e', we stop after appending it
                role_idx = self._sample_from_logits(logits_role[0], temp=temp)
                role_letter = self.role_index_to_letter[role_idx]
                if role_letter == 'e':
                    # We still need a hold for the end marker; sample it
                    hold_idx = self._sample_from_logits(logits_hold[0], temp=temp)
                    result.append((hold_idx, role_letter))
                    break
                # Otherwise sample hold and continue
                hold_idx = self._sample_from_logits(logits_hold[0], temp=temp)
                result.append((hold_idx, role_letter))
                # Append to context
                holds_ctx = torch.cat([holds_ctx, torch.tensor([[hold_idx]], device=self.device)], dim=1)
                roles_ctx = torch.cat([roles_ctx, torch.tensor([[role_idx]], device=self.device)], dim=1)

        return result
