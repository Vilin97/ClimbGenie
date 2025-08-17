import sqlite3, pandas as pd, re, numpy as np

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
    
    # Make holds_xy hold only (x, y) and add a new column 'roles' as a string like "sffmmfme"
    new_holds_xy_list = []
    new_roles_list = []
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

        # Recreate holds_xy, holds, holds_indices, and roles from sorted holds
        new_holds_xy = [(h['x'], h['y']) for h in holds_with_props]
        new_roles = "".join([h['role'] for h in holds_with_props])
        new_holds = "p".join([f"{h['id']}r{h['role_id']}" for h in holds_with_props])
        new_holds_indices = [hold_id_to_index(h['id']) for h in holds_with_props]

        new_holds_xy_list.append(str(new_holds_xy))
        new_roles_list.append(new_roles)
        new_holds_list.append(new_holds)
        new_holds_indices_list.append(new_holds_indices)

    df["holds_xy"] = new_holds_xy_list
    df["roles"] = new_roles_list
    df["holds"] = new_holds_list
    df["holds_indices"] = new_holds_indices_list

    df.to_csv(f"climbs_{split}.csv", index=False)