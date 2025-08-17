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
import src.utils as utils
import src.vis as vis
import pandas as pd
import numpy as np

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
