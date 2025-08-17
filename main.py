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
import src.vis as vis
import pandas as pd
import numpy as np

#%%
# Visualize some climbs
train_df = pd.read_csv("climbs_train.csv")
for (idx, row) in train_df.iloc[:2].iterrows():
    holds_xy = eval(row["holds_xy"])
    holds_indices = range(len(holds_xy))
    vis.draw_climb(holds_xy, title=f"{row['name']} | {row['grade']} @ {int(row['angle'])}° | {int(row['ascents'])} ascents", annotations=holds_indices)

#%%
train_df = pd.read_csv("climbs_train.csv")

cooc_gen = CooccurrenceGenerator()
cooc_gen.train(train_df)
# Convert stored logits to probabilities for visualization
_logits = cooc_gen.model["logits_trans"]  # torch.Tensor
M = np.exp(_logits.detach().cpu().numpy())
M = M / np.clip(M.sum(axis=1, keepdims=True), 1e-8, None)
# visualize the co-occurrence matrix
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 300
import seaborn as sns

plt.figure(figsize=(10, 8))
ax = sns.heatmap(M, cmap="viridis", square=True)
plt.title("Hold Co-occurrence Matrix Heatmap")
plt.xlabel("Hold Index")
plt.ylabel("Hold Index")
plt.tight_layout()

# Find the k highest values and annotate them
k = 3
flat_indices = np.argpartition(M.flatten(), -k)[-k:]
top_coords = [np.unravel_index(idx, M.shape) for idx in flat_indices]
for i, j in top_coords:
    ax.annotate(f"({i},{j})", xy=(j+0.5, i+0.5), color="red", ha="center", va="center", fontsize=7, fontweight="bold")

plt.show()
#%%
# Generate a random climb
climb_length = 10
for temp in [0.1, 0.4, 1, 10]:
    climb = cooc_gen.generate(length=climb_length, temp=temp)
    # visualize using generator's index_to_xy helper
    holds_xy = [(*cooc_gen.index_to_xy(hold), 'm') for hold in climb]
    vis.draw_climb(holds_xy, title=f"Generated Climb | temp={temp}", annotations=climb)

#%%
# Bigram model
train_df = pd.read_csv("climbs_train.csv")

bigram_gen = BigramGenerator()
bigram_gen.train(train_df)

# --- example: sample & visualize with your existing vis.draw_climb -------------
climb_length = 12
for temp in [0.5, 1.0, 2.0]:
    seq = bigram_gen.generate(length=climb_length, temp=temp)
    holds_xy = [(*bigram_gen.index_to_xy(h), 'm') for h in seq]
    vis.draw_climb(holds_xy, title=f"Bigram sample (temp={temp})", annotations=seq)

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
vis.draw_climb(holds_xy, title="AR sample (hold+role)", annotations=None)
# %%
