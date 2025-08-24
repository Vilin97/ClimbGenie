# %%
# only using the single baord setup: 
# LAYOUT_ID  = 1
# SIZE_ID    = 10
# SET_IDS    = (1, 20)

# Did to the splits, not present in kilter_splits.sqlite:
# 1. Filter out climbs with any hold ID > 4000 (~500 climbs total)
# 2. Filter out climbs with any role ID > 15 (~600 climbs total)
# 3. Fixed the holds_xy column to be a list of triples (x, y, role) with correct roles

# climbs_train.csv contains:
# uuid, name, difficulty, grade, angle, holds, holds_xy, ascents, quality, description
# holds.csv contains index, hold_id, x, y
# hold_roles.csv contains index, id, name, letter ('s', 'm', 'e', 'f'), color

#%%
from src.generator import CooccurrenceGenerator, BigramGenerator
from src.model import BigramModel
import src.utils as utils
import src.vis as vis
import pandas as pd

roles_df = pd.read_csv("hold_roles.csv")
def role_index_to_letter(idx):
    row = roles_df[roles_df["index"] == idx]
    assert len(row) == 1
    return row["letter"].item()

holds_df = pd.read_csv("holds.csv")
N_HOLDS = len(list(holds_df["hold_id"]))
def hold_index_to_xy(idx):
    row = holds_df[holds_df["index"] == idx]
    assert len(row) == 1
    return (row["x"].item(), row["y"].item())

def tokens_to_holds_and_roles(tokens):
    holds_xy = []
    roles = []
    for t in tokens:
        if t == 0:
            continue
        hold = (t - 1) % N_HOLDS
        role = (t - 1) // N_HOLDS
        print(t, hold, role)
        holds_xy.append(hold_index_to_xy(hold))
        roles.append(role_index_to_letter(role))
    return holds_xy, roles

#%%
bigram_model = BigramModel(num_difficulty_bins=1, num_angle_bins=1)
bigram_model.train()

#%%
train_df = pd.read_csv("climbs_train.csv")
val_df = pd.read_csv("climbs_val.csv")
train_nll = bigram_model.nll(train_df.sample(1000))
val_nll = bigram_model.nll(val_df.sample(1000))
print(f"Train NLL: {train_nll:.4f} nats/token")
print(f"Val   NLL: {val_nll:.4f} nats/token")

#%%
difficulty = 22 # V6
angle = 40
temperature = 1.0
tokens = bigram_model.generate(difficulty, angle, temperature)
holds_xy, roles = tokens_to_holds_and_roles(tokens)
vis.draw_climb(holds_xy, roles, title=f"Bigram sample | diff={difficulty}, angle={angle}, temp={temperature}")


#%%
# Visualize some climbs
train_df = pd.read_csv("climbs_train.csv")
for (idx, row) in train_df.iloc[:2].iterrows():
    holds_xy = eval(row["holds_xy"])
    roles = row["roles"]
    holds_indices = range(1,1+len(holds_xy))
    vis.draw_climb(holds_xy, roles, title=vis.make_title(row), annotations=holds_indices)

#%%
# Co-occurrence model
train_df = pd.read_csv("climbs_train.csv")

cooc_gen = CooccurrenceGenerator()
cooc_gen.train(train_df)
climb_length = 12
for temp in [0.5, 1.0, 2.0]:
    climb = cooc_gen.generate(length=climb_length, temp=temp)
    holds_xy = [cooc_gen.index_to_xy(hold) for hold in climb]
    roles = utils.infer_roles(climb, holds_xy)
    vis.draw_climb(holds_xy, roles, title=f"Co-occurence sample | temp={temp}", annotations=climb)

#%%
# Bigram model
train_df = pd.read_csv("climbs_train.csv")

bigram_gen = BigramGenerator()
bigram_gen.train(train_df)
climb_length = 12
for temp in [0.5, 1.0, 2.0]:
    seq = bigram_gen.generate(length=climb_length, temp=temp)
    holds_xy = [bigram_gen.index_to_xy(hold) for hold in seq]
    roles = utils.infer_roles(seq, holds_xy)
    vis.draw_climb(holds_xy, roles, title=f"Bigram Sample | temp={temp}", annotations=seq)

#%%
"AutoRegressive model (hold + role) — train briefly and sample"
import pandas as pd
import src.generator as gen
import importlib
importlib.reload(gen)
# AutoRegressive model (hold + role) — train briefly and sample
train_df = pd.read_csv("climbs_train.csv")
ar_gen = gen.AutoRegressiveGenerator(context_len=40, d_model=64, hidden_size=128)
# Light training for demo; increase epochs for better results
ar_gen.train(train_df, epochs=1, max_samples=200, verbose=True)

#%%
seq = ar_gen.generate(max_length=20, temp=1.0)
# seq is list of (hold_idx, role_letter)
holds_xy = [(*ar_gen.index_to_xy(h), r) for (h, r) in seq]
vis.draw_climb(holds_xy, title="AR sample (hold+role)", annotations=range(1,1+len(seq)))
# %%
