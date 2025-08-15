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
    
    # Make holds_xy hold a list of triples (x, y, role) and reorder holds
    new_holds_xy_list = []
    new_holds_list = []
    new_holds_indices_list = []

    for _, row in df.iterrows():
        holds_with_props = []
        for hold_id, role_id in parse_holds(row["holds"]):
            pos = holes_pos.get(hold_id)
            letter = role_id_to_letter.get(role_id)
            if pos and letter:
                holds_with_props.append({"id": hold_id, "x": pos[0], "y": pos[1], "role": letter, "role_id": role_id})

        def sort_key(hold):
            if hold['role'] == 's':
                return (0, hold['y'], hold['x'])
            elif hold['role'] == 'e':
                return (2, hold['y'], hold['x'])
            else: # 'm' or 'f'
                return (1, hold['y'], hold['x'])

        holds_with_props.sort(key=sort_key)

        # Recreate holds_xy, holds, and holds_indices from sorted holds
        new_holds_xy = [(h['x'], h['y'], h['role']) for h in holds_with_props]
        new_holds = "p".join([f"{h['id']}r{h['role_id']}" for h in holds_with_props])
        new_holds_indices = [hold_id_to_index(h['id']) for h in holds_with_props]

        new_holds_xy_list.append(str(new_holds_xy))
        new_holds_list.append(new_holds)
        new_holds_indices_list.append(new_holds_indices)

    df["holds_xy"] = new_holds_xy_list
    df["holds"] = new_holds_list
    df["holds_indices"] = new_holds_indices_list

    df.to_csv(f"climbs_{split}.csv", index=False)

#%%
# --- 2. make a simple co-occurence model ---
from src.generator import CooccurrenceGenerator, BigramGenerator
import src.vis as vis
import importlib

train_df = pd.read_csv("climbs_train.csv")

cooc_gen = CooccurrenceGenerator()
cooc_gen.train(train_df)
M = cooc_gen.model[0]

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
train_df = pd.read_csv("climbs_train.csv")
top_pairs = [(170, 119), (459, 312)]
def contains_top_pair(holds_indices):
    holds = set(holds_indices)
    return any(a in holds and b in holds for a, b in top_pairs)

filtered_df = train_df[train_df["holds_indices"].apply(lambda x: contains_top_pair(eval(x)))]

import importlib
import src.vis as vis
importlib.reload(vis)
for (idx, row) in filtered_df.reset_index(drop=True).iterrows():
    # vis.draw_climb_row(row)
    holds_xy = eval(row["holds_xy"])
    holds_indices = range(len(holds_xy))
    vis.draw_climb(holds_xy, title=f"{row['name']} | {row['grade']} @ {int(row['angle'])}° | {int(row['ascents'])} ascents", annotations=holds_indices)
    if idx >= 4:
        break

#%%
# Generate a random climb
climb_length = 10
for temp in [0.1, 0.4, 1, 10]:
    climb = cooc_gen.generate(length=climb_length, temp=temp)

    # visualize
    holds_xy = [(cooc_gen.holes_pos[cooc_gen.index_to_hold_id(hold)][0], cooc_gen.holes_pos[cooc_gen.index_to_hold_id(hold)][1], 'm') for hold in climb]
    vis.draw_climb(holds_xy, title=f"Generated Climb | temp={temp}", annotations=climb)

#%%
# Bigram model
train_df = pd.read_csv("climbs_train.csv")

bigram_gen = BigramGenerator()
bigram_gen.train(train_df)

# --- example: sample & visualize with your existing vis.draw_climb -------------
importlib.reload(vis)

climb_length = 12
for temp in [0.5, 1.0, 2.0]:
    seq = bigram_gen.generate(length=climb_length, temp=temp)
    holds_xy = [(bigram_gen.holes_pos[bigram_gen.index_to_hold_id(h)][0], bigram_gen.holes_pos[bigram_gen.index_to_hold_id(h)][1], 'm') 
                for h in seq]
    vis.draw_climb(holds_xy, title=f"Bigram sample (temp={temp})", annotations=seq)