import numpy as np
import pandas as pd
import random
import re
import ast

class Generator:
    def __init__(self):
        self.model = None
        self.all_hold_ids = None
        self.holes_pos = None
        self.role_id_to_letter = None
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

    def _parse_holds_indices(self, holds_indices_str: str) -> list[int]:
        return ast.literal_eval(holds_indices_str)

    def _load_hold_data(self):
        holds_df = pd.read_csv("holds.csv")
        self.all_hold_ids = list(holds_df["hold_id"])
        self.holes_pos = dict(zip(holds_df["hold_id"], zip(holds_df["x"], holds_df["y"])))
        
        roles_df = pd.read_csv("hold_roles.csv")
        self.role_id_to_letter = dict(zip(roles_df["id"], roles_df["letter"]))

    def hold_id_to_index(self, hold_id: int) -> int:
        return self.all_hold_ids.index(hold_id)

    def index_to_hold_id(self, idx: int) -> int:
        return self.all_hold_ids[idx]

class CooccurrenceGenerator(Generator):
    def train(self, df: pd.DataFrame):
        co_occurrence_matrix = np.zeros((len(self.all_hold_ids), len(self.all_hold_ids)))
        hold_counts = np.ones(len(self.all_hold_ids))

        for holds_str in df["holds_indices"]:
            indices_in_climb = self._parse_holds_indices(holds_str)
            for i in indices_in_climb:
                hold_counts[i] += 1
                for j in indices_in_climb:
                    if j != i:
                        co_occurrence_matrix[i, j] += 1
        
        M = co_occurrence_matrix / hold_counts[:, np.newaxis]
        self.model = (M, hold_counts)

    def generate(self, length: int, temp: float = 1.0, angle: int = None, difficulty: str = None):
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        M, hold_counts = self.model
        
        climb = [self._pick(hold_counts, temp)]
        for _ in range(length - 1):
            climb.append(self._pick(M[climb[-1]], temp))
        return climb

class BigramGenerator(Generator):
    def train(self, df: pd.DataFrame, alpha: float = 1.0):
        counts = np.full((len(self.all_hold_ids), len(self.all_hold_ids)), alpha, dtype=np.float64)
        start_counts = np.full(len(self.all_hold_ids), alpha, dtype=np.float64)

        for _, row in df.iterrows():
            seq = self._parse_holds_indices(row["holds_indices"])
            if not seq: continue
            start_counts[seq[0]] += 1
            for a, b in zip(seq, seq[1:]):
                counts[a, b] += 1

        M = counts / counts.sum(axis=1, keepdims=True)
        P0 = start_counts / start_counts.sum()
        self.model = (M, P0)

    def generate(self, length: int, temp: float = 1.0, angle: int = None, difficulty: str = None):
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() first.")
        M, P0 = self.model

        climb = [self._pick(P0, temp)]
        for _ in range(length - 1):
            climb.append(self._pick(M[climb[-1]], temp))
        return climb
