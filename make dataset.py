#%%
"Read schema"
import sqlite3
conn = sqlite3.connect("kilter_dataset.csv")
tables = ["climbs", "climb_stats", "climb_cache_fields",
          "difficulty_grades", "beta_links", "placements", "placement_roles", "holes"]

for t in tables:
    print(f"\n-- {t} --")
    for cid, name, typ, notnull, dflt, pk in conn.execute(f"PRAGMA table_info({t});"):
        print(f"{name:<25} {typ}")
conn.close()
#%%
import sqlite3, pandas as pd

DB        = "kilter_dataset.csv"
NEW_TABLE = "climbs_l1_s10"
LAYOUT_ID = 1
SIZE_ID   = 10
SET_IDS   = (1, 20)
TH        = 0.999                    # 99.9 %

# ── helpers ──────────────────────────────────────────────────────────────
_frames_to_ids = lambda fr: {int(seg.split("r")[0]) for seg in fr.split("p") if seg}

def decode_frames(frames: str):
    """
    Return a list of (x, y, role_code) tuples where role_code ∈ {S, E, T, I}:
        S  start
        E  end / finish
        T  toe  (foot-only hold)
        I  intermediate / everything else
    """
    if pd.isna(frames):
        return []
    out = []
    for seg in frames.split("p"):
        if not seg:
            continue
        pid = int(seg.split("r")[0])
        if pid not in placements.index:
            continue
        r = placements.loc[pid]
        role = r["role_name"]
        if role == "start":
            code = "S"
        elif role == "finish":
            code = "E"         # ← finish = E
        elif role == "foot":
            code = "T"
        else:
            code = "I"
        out.append((r["x"], r["y"], code))
    return out

def holds_stats(coords):
    """
    From a list[(x, y, code)]  →  (#holds, has_start, has_finish, diameter)
    """
    n = len(coords)
    if n == 0:
        return 0, False, False, 0.0
    has_start  = any(c == "S" for _, _, c in coords)
    has_finish = any(c == "E" for _, _, c in coords)   # ← uses E
    diam = 0.0
    for i in range(n):
        xi, yi, _ = coords[i]
        for j in range(i + 1, n):
            xj, yj, _ = coords[j]
            d = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            if d > diam:
                diam = d
    return n, has_start, has_finish, diam

def sf_too_close(coords, thresh=3):
    """
    True  ➜  every start/finish pair is < thresh Manhattan units apart
    """
    starts   = [(x, y) for x, y, c in coords if c == "S"]
    finishes = [(x, y) for x, y, c in coords if c == "E"]   # ← uses E
    if not starts or not finishes:
        return True
    for sx, sy in starts:
        for fx, fy in finishes:
            if abs(sx - fx) + abs(sy - fy) >= thresh:
                return False
    return True

def num_holds(frames: str) -> int:
    if pd.isna(frames):
        return 0
    return sum(1 for seg in frames.split("p") if seg)

def has_valid_start_finish(frame_str: str) -> bool:
    placements = pd.read_sql_query("""
    SELECT p.id   AS placement_id,
           pr.name AS role_name,
           h.x, h.y
    FROM placements p
    JOIN holes           h  ON h.id = p.hole_id
    JOIN placement_roles pr ON pr.id = p.role_id
    """, conn).set_index("placement_id")
    if pd.isna(frame_str):
        return False
    pids = [int(seg.split("r")[0]) for seg in frame_str.split("p") if seg]
    starts  = [(placements.loc[pid, "x"], placements.loc[pid, "y"])
               for pid in pids
               if pid in placements.index and placements.loc[pid, "role_name"] == "start"]
    finishes = [(placements.loc[pid, "x"], placements.loc[pid, "y"])
                for pid in pids
                if pid in placements.index and placements.loc[pid, "role_name"] == "finish"]

    if not starts or not finishes:
        return False                          # missing either start or finish

    # Manhattan distance between every start/finish pair
    for sx, sy in starts:
        for fx, fy in finishes:
            if abs(sx - fx) + abs(sy - fy) < 3:
                return False                  # too close
    return True

def inside_ids(conn):
    l,r,b,t = conn.execute(
        "SELECT edge_left,edge_right,edge_bottom,edge_top "
        "FROM product_sizes WHERE id=?", (SIZE_ID,)).fetchone()
    rows = conn.execute(f"""
        SELECT p.id
        FROM placements p
        JOIN holes h ON h.id = p.hole_id
        WHERE p.layout_id = ?
          AND p.set_id    IN ({','.join('?'*len(SET_IDS))})
          AND h.x BETWEEN ? AND ? AND h.y BETWEEN ? AND ?""",
        (LAYOUT_ID,*SET_IDS,l,r,b,t))
    return {pid for (pid,) in rows}

def first_notnull(*vals):
    for v in vals:
        if pd.notna(v):
            return v
    return None

# ── build ───────────────────────────────────────────────────────────────
with sqlite3.connect(DB) as conn:
    board_ids = inside_ids(conn)

    climbs_all = pd.read_sql_query(
        "SELECT * FROM climbs WHERE layout_id=? AND is_draft=0",
        conn, params=(LAYOUT_ID,))
    fits = climbs_all["frames"].apply(lambda fr: _frames_to_ids(fr).issubset(board_ids))
    climbs = climbs_all[fits].copy()
    uuids  = tuple(climbs["uuid"])

    conn.execute("DROP TABLE IF EXISTS tmp_u")
    conn.execute("CREATE TEMP TABLE tmp_u(id TEXT PRIMARY KEY)")
    conn.executemany("INSERT INTO tmp_u(id) VALUES (?)", [(u,) for u in uuids])

    stats = pd.read_sql_query(
        "SELECT * FROM climb_stats cs JOIN tmp_u ON cs.climb_uuid = tmp_u.id", conn)

    cache = pd.read_sql_query(
        "SELECT * FROM climb_cache_fields cf JOIN tmp_u ON cf.climb_uuid = tmp_u.id", conn)

    beta_links = pd.read_sql_query(
        """
        SELECT climb_uuid, link, angle, foreign_username
        FROM beta_links
        WHERE climb_uuid IN (SELECT id FROM tmp_u)
          AND is_listed = 1
          AND link LIKE 'https://www.instagram.com%'""", conn)

    beta_agg = (beta_links.groupby("climb_uuid", as_index=False)
                .agg(beta_links      = ("link",  lambda s: "|".join(s.dropna())),
                     beta_angles     = ("angle", lambda s: "|".join(s.dropna().astype(str))),
                     beta_usernames  = ("foreign_username", lambda s: "|".join(s.dropna()))))

    # rows WITH stats
    df_stats = (stats.rename(columns={"angle":"angle_adjustable"})
                .merge(climbs, how="left", left_on="climb_uuid", right_on="uuid",
                       suffixes=("", "_src"))
                .merge(cache, how="left", on="climb_uuid")
                .merge(beta_agg, how="left", on="climb_uuid"))

    # rows WITHOUT stats
    missing = set(uuids) - set(stats["climb_uuid"])
    if missing:
        cb = climbs[climbs["uuid"].isin(missing)].rename(columns={"angle":"angle_set"})
        ca = cache[cache["climb_uuid"].isin(missing)]
        df_missing = (cb.merge(ca, how="left", left_on="uuid", right_on="climb_uuid")
                        .merge(beta_agg, how="left", left_on="uuid", right_on="climb_uuid"))
        df_missing["angle_adjustable"] = pd.NA
    else:
        df_missing = pd.DataFrame(columns=df_stats.columns)

    df = pd.concat([df_stats, df_missing], ignore_index=True, sort=False)

    if "angle_set" not in df.columns:
        df["angle_set"] = pd.NA
    df["angle"] = df.apply(lambda r: first_notnull(r["angle_adjustable"], r["angle_set"]), axis=1)

    # -- difficulty per angle:  avg first, then display_difficulty
    def choose_diff(r):
        return first_notnull(r.get("difficulty_average"),
                             r.get("display_difficulty"),
                             r.get("display_difficulty_y"))
    df["difficulty_numeric"] = df.apply(choose_diff, axis=1)

    grades = pd.read_sql_query(
        "SELECT difficulty, boulder_name AS boulder_grade, route_name AS route_grade "
        "FROM difficulty_grades", conn)
    df = df.merge(grades, how="left",
                  left_on=df["difficulty_numeric"].round().astype("Int64"),
                  right_on="difficulty").drop(columns=["difficulty"])

    # -------- tidy columns ----------------------------------------------
    df = (df.sort_values(["uuid","angle"])
            .drop_duplicates(subset=["uuid","angle"], keep="first"))

    rename_keep = {"ascensionist_count_x":"ascensionist_count",
                   "quality_average_x"   :"quality_average"}
    dupes = [c for c in df.columns if c.endswith(("_x","_y"))]
    df = df.drop(columns=[c for c in dupes if c not in rename_keep]).rename(columns=rename_keep)
    df = df.drop(columns=[c for c in df.columns if c.startswith("climb_uuid")])
    df = df.loc[:, ~df.columns.duplicated()]

    must_keep = {"uuid","angle","angle_adjustable","angle_set","name",
                 "difficulty_numeric","boulder_grade","route_grade",
                 "ascensionist_count","frames","description","is_nomatch","layout_id","product_size_id","set_id"}
    for col in df.columns.difference(must_keep):
        s = df[col]
        if s.isna().mean() >= TH:
            df = df.drop(columns=col)
        else:
            vc = s.value_counts(dropna=False)
            top_freq = (vc.iloc[0] / len(s)) if len(vc) else 0
            if top_freq >= TH:
                df = df.drop(columns=col)

    placements = pd.read_sql_query("""
    SELECT p.id   AS placement_id,
           pr.name AS role_name,
           h.x, h.y
    FROM placements p
    JOIN holes           h  ON h.id = p.hole_id
    LEFT JOIN placement_roles pr
           ON pr.id = p.default_placement_role_id
    """, conn).set_index("placement_id")

    # ----- compute per-row features --------------------------
    decoded = df["frames"].apply(decode_frames)
    stats   = decoded.apply(holds_stats)
    df["num_holds"]  = stats.apply(lambda t: t[0])
    df["has_start"]  = stats.apply(lambda t: t[1])
    df["has_finish"] = stats.apply(lambda t: t[2])
    df["diameter"]   = stats.apply(lambda t: t[3])
    df["frames_xy"]  = decoded.apply(
        lambda lst: "|".join(f"({x},{y}){c}" for x,y,c in lst))

    # ----- filters -------------------------------------------
    df = df[df["ascensionist_count"] > 0]      # > 0 ascents
    df = df[df["difficulty_numeric"] <= 31]  # ≤ V14
    df = df[df["num_holds"] >= 3]              # ≥ 3 holds
    df = df[df["has_start"]]                   # must have a start
    df = df[df["num_holds"] <= 40]             # sanity cap (optional)

    # ----- keep / order columns ------------------------------
    core = ["uuid","angle","name",
            "difficulty_numeric","boulder_grade","route_grade",
            "ascensionist_count","diameter","num_holds",
            "angle_adjustable","angle_set",
            "frames_xy","frames","description","is_nomatch"]
    df = df[[c for c in core if c in df.columns] +
            [c for c in df.columns if c not in core]]

    # sort by popularity then angle
    df = df.sort_values(["ascensionist_count","angle"],
                        ascending=[False, True], na_position="last")


    # write table + unique index
    df.to_sql(NEW_TABLE, conn, if_exists="replace", index=False)
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {NEW_TABLE}_pk "
                 f"ON {NEW_TABLE}(uuid, angle)")
    print(f"✓ {len(df):,} rows × {df.shape[1]} columns  →  {NEW_TABLE}")

#%%
"Split into train/val/test sets based on UUIDs"
# split_climbs_into_splits.py
import sqlite3, random, pandas as pd
from pathlib import Path

SOURCE_DB     = Path("kilter_dataset.csv")   # original DB
SOURCE_TABLE  = "climbs_l1_s10"

TARGET_DB     = Path("kilter_splits.sqlite") # new DB file to create
TRAIN_TABLE   = "kilter_train"
VAL_TABLE     = "kilter_val"
TEST_TABLE    = "kilter_test"

RNG_SEED      = 42                           # reproducible shuffle

# ----------------------------------------------------------------------
# 1. collect UUIDs
# ----------------------------------------------------------------------
with sqlite3.connect(SOURCE_DB) as src:
    uuids = [u for (u,) in src.execute(
        f"SELECT DISTINCT uuid FROM {SOURCE_TABLE}")]
random.Random(RNG_SEED).shuffle(uuids)

n_total = len(uuids)
n_train = int(n_total * 0.80)
n_val   = int(n_total * 0.10)

train_uuids = set(uuids[:n_train])
val_uuids   = set(uuids[n_train:n_train + n_val])
test_uuids  = set(uuids[n_train + n_val:])

splits = [(TRAIN_TABLE, train_uuids),
          (VAL_TABLE,   val_uuids),
          (TEST_TABLE,  test_uuids)]

# ----------------------------------------------------------------------
# 2. create new database and write split tables
# ----------------------------------------------------------------------
if TARGET_DB.exists():
    TARGET_DB.unlink()                       # start fresh

with sqlite3.connect(SOURCE_DB) as src, \
     sqlite3.connect(TARGET_DB)   as dst:

    for table_name, uuid_set in splits:
        print(f"building {table_name} …")

        # fetch all rows for this split into a DataFrame
        ph = ",".join("?" * len(uuid_set))
        df = pd.read_sql_query(
            f"SELECT * FROM {SOURCE_TABLE} WHERE uuid IN ({ph})",
            src, params=list(uuid_set))

        # write to the new DB
        df.to_sql(table_name, dst, index=False)

        print(f"  → {len(df):,} rows, {len(uuid_set):,} unique climbs")

print("\n✓ finished — splits are stored in", TARGET_DB.resolve())

# ----------------------------------------------------------------------
# copy support tables (everything from the schema list except `climbs`)
# ----------------------------------------------------------------------
SUPPORT_TABLES = [
    "difficulty_grades",
    "placements",
    "placement_roles",
    "holes",
]

with sqlite3.connect(SOURCE_DB) as src, \
     sqlite3.connect(TARGET_DB) as dst:

    for t in SUPPORT_TABLES:
        # drop if exists in target to keep it idempotent
        dst.execute(f"DROP TABLE IF EXISTS {t}")

        # copy schema
        schema_sql = src.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name=?", (t,)).fetchone()[0]
        dst.execute(schema_sql)

        # copy data in chunks to avoid memory spikes
        CHUNK = 50_000
        offset = 0
        while True:
            rows = src.execute(f"SELECT * FROM {t} LIMIT {CHUNK} OFFSET {offset}").fetchall()
            if not rows:
                break
            placeholders = ",".join("?" * len(rows[0]))
            dst.executemany(f"INSERT INTO {t} VALUES ({placeholders})", rows)
            offset += CHUNK

        count = dst.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"copied {t:24s} → {count:,} rows")

print("\n✓ support tables copied to", TARGET_DB.resolve())

#%%
"Make histograms of # ascends and # holds per climb"

import sqlite3, pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DB = "kilter_dataset.csv"
TABLE = "climbs_l1_s10"

# ---------------- load per-angle rows ----------------
with sqlite3.connect(DB) as conn:
    per_angle = pd.read_sql_query(
        f"SELECT uuid, ascensionist_count, frames FROM {TABLE}", conn)

# ---------------- aggregate per uuid -----------------
agg = (per_angle
       .groupby("uuid", as_index=False)
       .agg(total_ascensionists=("ascensionist_count", "sum"),
            frames=("frames", "first")))

def n_holds(frames: str) -> int:
    if pd.isna(frames):
        return 0
    return sum(1 for seg in frames.split("p") if seg)

agg["num_holds"] = agg["frames"].apply(n_holds)

# ---------------- stats requested --------------------
zero_asc = (agg["total_ascensionists"] == 0).sum()
few_holds = (agg["num_holds"] < 3).sum()
print(f"Climbs with 0 ascensionists : {zero_asc:,}")
print(f"Climbs with <3 holds        : {few_holds:,}")

# ---------------- clamp for plots --------------------
asc_clamped   = np.clip(agg["total_ascensionists"], None, 20)
holds_clamped = np.clip(agg["num_holds"], None, 50)

# ---------------- histogram 1 ------------------------
fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=150)
ax1.hist(asc_clamped, bins=50)
ax1.set_xlabel("Total ascensionists (clamped at 20)")
ax1.set_ylabel("# climbs")
ax1.set_title("Distribution of ascensionists per climb (summed across angles)")
plt.tight_layout()

# ---------------- histogram 2 ------------------------
fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=150)
ax2.hist(holds_clamped, bins=range(0, 51))
ax2.set_xlabel("# holds in climb (clamped at 50)")
ax2.set_ylabel("# climbs")
ax2.set_title("Distribution of hold counts per climb")
plt.tight_layout()

plt.show()
#%%
import sqlite3, random, pandas as pd
import draw                       # assumes `draw.py` in the user’s env

# ---------------- configuration ----------------
DB         = "kilter_dataset.csv"
TABLE      = "climbs_l1_s10"
LAYOUT_ID  = 1
SIZE_ID    = 10
SET_IDS    = (1, 20)

N_DRAW     = 5           # how many random climbs to show
RNG_SEED   = None         # set to an int for reproducible draws

# ---------------- load uuids -------------------
with sqlite3.connect(DB) as conn:
    uuids = [u for (u,) in conn.execute(f"SELECT DISTINCT uuid FROM {TABLE}")]

if RNG_SEED is not None:
    random.seed(RNG_SEED)

random_uuids = random.sample(uuids, min(N_DRAW, len(uuids)))

# fetch metadata for printed summary
with sqlite3.connect(DB) as conn:
    meta_df = pd.read_sql_query(
        f"""
        SELECT uuid, name, angle, boulder_grade,
               num_holds, diameter, ascensionist_count
        FROM {TABLE}
        WHERE uuid IN ({','.join('?'*len(random_uuids))})
        """,
        conn, params=random_uuids)

# ------------- draw & print --------------------
for u in random_uuids:
    rows = meta_df[meta_df["uuid"] == u]
    meta = rows.iloc[0]

    # draw the climb
    draw.draw_climb(
        u,
        LAYOUT_ID,
        SIZE_ID,
        SET_IDS,
        title=f"{meta['name']}  (uuid {u[:6]}…)"
    )

    # print a short summary
    angle_grade = ", ".join(
        f"{int(r.angle)}°: {r.boulder_grade}"
        for r in rows.itertuples()
    )
    print(f"\n• {meta['name']}")
    print(f"  uuid: {u}")
    print(f"  grades @ angles: {angle_grade}")
    print(f"  holds: {int(meta.num_holds)} | diameter: {meta.diameter:.1f}  | ascents: {int(meta.ascensionist_count)}")

