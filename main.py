# %%
# only using the single baord setup: 
# LAYOUT_ID  = 1
# SIZE_ID    = 10
# SET_IDS    = (1, 20)

# Did to the splits, not present in kilter_splits.sqlite:
# 1. Filter out climbs with any hold ID > 4000 (~500 climbs total)
# 2. Filter out climbs with any role ID > 15 (~600 climbs total)
# 3. Fixed the holds_xy column to be a list of triples (x, y, role) with correct roles

# holds.csv contains index, hold_id, x, y
# hold_roles contains index, id, name, letter ('s', 'm', 'e', 'f'), color

import sqlite3, pandas as pd, re, requests, io, matplotlib.pyplot as plt
from PIL import Image

#%%
DB = "kilter_splits.sqlite"
LAYOUT_ID = 1
SIZE_ID = 10
SET_IDS = [1, 20]
API = "https://api.kilterboardapp.com/img/{}"

# --- 1. Make DataFrame & save as CSV ---
def extract_hold_ids(frames_col):
    # frames_col: string like "1096r1p1102r2p..."
    return [int(seg.split("r")[0]) for seg in str(frames_col).split("p") if seg]

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
    df.to_csv(f"climbs_{split}.csv", index=False)

#%%
# --- 2. Hold name/index conversion using climbs_train.csv ---

# Load all unique hold IDs from climbs_train.csv
train_df = pd.read_csv("climbs_train.csv")

all_hold_ids = sorted({hid for holds in train_df["holds"] for hid in extract_hold_ids(holds)})

def hold_id_to_index(hold_id: int) -> int:
    """Convert hold ID (e.g., 1073) to index (0...475)"""
    return all_hold_ids.index(hold_id)

def index_to_hold_id(idx: int) -> int:
    """Convert index (0...475) to hold ID (e.g., 1073)"""
    return all_hold_ids[idx]

for i in range(len(all_hold_ids)):
    assert hold_id_to_index(index_to_hold_id(i)) == i
