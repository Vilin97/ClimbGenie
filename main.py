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
import sqlite3, pandas as pd, re, numpy as np

def extract_hold_ids(frames_col):
    # frames_col: string like "1096r1p1102r2p..."
    return [int(seg.split("r")[0]) for seg in str(frames_col).split("p") if seg]

holds_df = pd.read_csv("holds.csv")
all_hold_ids = list(holds_df["hold_id"])

def hold_id_to_index(hold_id: int) -> int:
    """Convert hold ID (e.g., 1073) to index (0...N) using holds.csv order"""
    return all_hold_ids.index(hold_id)

def index_to_hold_id(idx: int) -> int:
    """Convert index (0...N) to hold ID (e.g., 1073) using holds.csv order"""
    return all_hold_ids[idx]

for i in range(len(all_hold_ids)):
    assert hold_id_to_index(index_to_hold_id(i)) == i

#%%
DB = "kilter_splits.sqlite"
LAYOUT_ID = 1
SIZE_ID = 10
SET_IDS = [1, 20]
API = "https://api.kilterboardapp.com/img/{}"

# --- 1. Make DataFrame & save as CSV ---
def parse_holds(holds_str):
    # returns list of (hold_id, role_id)
    return [
        (int(seg.split("r")[0]), int(re.search(r"r(\d+)", seg).group(1)))
        for seg in str(holds_str).split("p")
        if seg and re.search(r"r(\d+)", seg)
    ]

roles_df = pd.read_csv("hold_roles.csv")
role_id_to_letter = dict(zip(roles_df["id"], roles_df["letter"]))

holds_df = pd.read_csv("holds.csv")
holes_pos = dict(zip(holds_df["hold_id"], zip(holds_df["x"], holds_df["y"])))

for split in ["train", "val", "test"]:
    with sqlite3.connect(DB) as conn:
        df = pd.read_sql_query(
            f"SELECT uuid, name, difficulty_numeric as difficulty, boulder_grade as grade, angle, frames as holds, frames_xy as holds_xy, ascensionist_count as ascents, quality_average as quality, description FROM kilter_{split}",
            conn
        )
    # Drop rows with any hold ID > 4000 (~500 climbs total)
    df = df[~df["holds"].apply(lambda h: any(hid > 4000 for hid in extract_hold_ids(h)))]
    # Drop rows with any role ID > 15 (~600 climbs total)
    df = df[~df["holds"].apply(lambda h: any(int(re.search(r"r(\d+)", seg).group(1)) > 15 for seg in str(h).split("p") if seg and re.search(r"r(\d+)", seg)))]
    
    # Make holds_xy hold a list of triples (x, y, role)
    new_holds_xy = []
    for _, row in df.iterrows():
        triples = []
        for hold_id, role_id in parse_holds(row["holds"]):
            pos = holes_pos.get(hold_id)
            letter = role_id_to_letter.get(role_id)
            if pos and letter:
                triples.append((pos[0], pos[1], letter))
        new_holds_xy.append(str(triples))
    df["holds_xy"] = new_holds_xy

    # save hold ids as a list of integers
    df["holds_indices"] = df["holds"].apply(lambda h: [hold_id_to_index(hold_id) for hold_id in extract_hold_ids(h)])

    df.to_csv(f"climbs_{split}.csv", index=False)

#%%
# --- 2. make a simple co-occurence model ---
train_df = pd.read_csv("climbs_train.csv")

# num_holds = 323 # focus on hand holds
num_holds = len(all_hold_ids)
co_occurrence_matrix = np.zeros((num_holds, num_holds))
hold_counts = np.ones(num_holds) # to avoid division by zero

for holds_str in train_df["holds_indices"]:
    indices_in_climb = [i for i in eval(holds_str) if i < num_holds]
    
    for i in indices_in_climb:
        hold_counts[i] += 1
        for j in indices_in_climb:
            if j != i:
                co_occurrence_matrix[i, j] += 1

# M_ij = P(hold_j | hold_i) = count(i and j) / count(i)
M = co_occurrence_matrix / hold_counts[:, np.newaxis]

#%%
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
# Visualize some climbs
top_pairs = [(170, 119), (459, 312)]
def contains_top_pair(holds_indices):
    holds = set(holds_indices)
    return any(a in holds and b in holds for a, b in top_pairs)

filtered_df = train_df[train_df["holds_indices"].apply(lambda x: contains_top_pair(eval(x)))]

import importlib
import src.vis as vis
importlib.reload(vis)
for (idx, row) in filtered_df.reset_index(drop=True).iterrows():
    vis.draw_climb_row(row)
    if idx >= 4:
        break

#%%
# Generate a random climb
import random

def sample_climb(hold_counts, co_occurence_matrix, length, temp):
    climb = []
    # sample the first hold
    weights = np.array(hold_counts) ** (1 / temp)
    hold = random.choices(range(num_holds), weights=weights, k=1)[0]
    climb.append(hold)

    # sample the next holds
    for _ in range(length - 1):
        weights = np.array(co_occurence_matrix[hold]) ** (1 / temp)
        next_hold = random.choices(range(num_holds), weights=weights, k=1)[0]
        climb.append(next_hold)
        hold = next_hold
    return climb

climb_length = 10
for temp in [0.1, 0.4, 1, 10]:
    climb = sample_climb(hold_counts, M, climb_length, temp)

    # visualize
    holds_xy = [(holes_pos[index_to_hold_id(hold)][0], holes_pos[index_to_hold_id(hold)][1], 'm') for hold in climb]
    vis.draw_climb(holds_xy, title=f"Generated Climb | temp={temp}", annotations=climb)