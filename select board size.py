#%%
import sqlite3, re
from collections import defaultdict
from typing import Dict, Tuple, List, Set

def board_stats(db_path: str = "kilter_dataset.csv") -> List[dict]:
    """
    Return one dictionary per board (layout_id, size_id):

        {
          "layout_id": 1,
          "size_id"  : 28,
          "sets"     : [1, 20],                  # all sets bolted on
          "holds_per_set": {1: 683, 20: 72},     # counts inside the board edges
          "holds_total"  : 755,                  # union of all sets
          "climbs"       : 1920                  # climbs fully contained
        }

    A climb is counted for *every* board whose union-of-sets completely
    contains its placement-ids.  Edge points are included (>=, <=).
    """

    # ── helpers ────────────────────────────────────────────────────────────
    _frames_to_ids = lambda f: {int(seg.split("r")[0]) for seg in f.split("p") if seg}

    with sqlite3.connect(db_path) as conn:
        # 1) geometry for each size_id
        edges: Dict[int, Tuple[float, float, float, float]] = {
            sid: (l, r, b, t)
            for sid, l, r, b, t in conn.execute(
                "SELECT id,edge_left,edge_right,edge_bottom,edge_top FROM product_sizes")
        }

        # 2) enumerate all (layout_id, size_id, set_id) rows
        rows = conn.execute(
            "SELECT layout_id, product_size_id, set_id "
            "FROM product_sizes_layouts_sets").fetchall()

        # 3) gather sets per board and count holds per set
        board_sets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        holds_per_set: Dict[Tuple[int,int,int], Set[int]] = {}

        for lid, sid, set_id in rows:
            board_sets[(lid, sid)].append(set_id)
            l, r, b, t = edges[sid]
            ids_inside = {
                pid for (pid,) in conn.execute("""
                    SELECT p.id
                    FROM placements p
                    JOIN holes h ON h.id = p.hole_id
                    WHERE p.layout_id = :lid AND p.set_id = :set_id
                      AND h.x >= :l AND h.x <= :r AND h.y >= :b AND h.y <= :t
                """, dict(lid=lid, set_id=set_id, l=l, r=r, b=b, t=t))
            }
            holds_per_set[(lid, sid, set_id)] = ids_inside

        # 4) union of holds for each board
        board_union: Dict[Tuple[int,int], Set[int]] = {}
        for (lid, sid), sets in board_sets.items():
            ids = set().union(*(holds_per_set[(lid, sid, s)] for s in sets))
            board_union[(lid, sid)] = ids

        # 5) count climbs
        climb_counts: Dict[Tuple[int,int], int] = defaultdict(int)
        for layout_id, frames in conn.execute(
            "SELECT layout_id, frames FROM climbs WHERE is_draft = 0"):
            f_ids = _frames_to_ids(frames)
            for (lid, sid), ids in board_union.items():
                if lid == layout_id and f_ids.issubset(ids):
                    climb_counts[(lid, sid)] += 1

    # 6) assemble result objects
    stats = []
    for (lid, sid), sets in board_sets.items():
        hps = {s: len(holds_per_set[(lid, sid, s)]) for s in sets}
        stats.append(dict(
            layout_id      = lid,
            size_id        = sid,
            sets           = sorted(sets),
            holds_per_set  = hps,
            holds_total    = sum(hps.values()),
            climbs         = climb_counts.get((lid, sid), 0)
        ))

    # nice ordering
    stats.sort(key=lambda d: (d["layout_id"], d["size_id"]))
    return stats

stats = board_stats()
for entry in stats:
    lid, sid = entry["layout_id"], entry["size_id"]
    print(f"layout {lid:2d}  size {sid:2d}  →  {entry['climbs']:5d} climbs"
          f"  |  holds {entry['holds_total']:4d}"
          f"  (per set: {entry['holds_per_set']})")

#%%
import sqlite3, io, requests, matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image
import matplotlib as mpl
import matplotlib.patheffects as pe

mpl.rcParams["figure.dpi"]   = 300   # on-screen
mpl.rcParams["savefig.dpi"]  = 300   # files

DB   = "kilter_dataset.csv"
API  = "https://api.kilterboardapp.com/img/{}"

# ── tiny helper ─────────────────────────────────────────────────────────────
_q = lambda c, sql, **p: c.execute(sql, p).fetchall()

# ── sql helpers ─────────────────────────────────────────────────────────────
def board_rows(conn, layout_id):
    return _q(conn, """
      SELECT product_size_id,set_id,image_filename
      FROM product_sizes_layouts_sets
      WHERE layout_id=:lid ORDER BY product_size_id,set_id""", lid=layout_id)

def edges(conn, sid):
    return _q(conn, "SELECT edge_left,edge_right,edge_bottom,edge_top "
                    "FROM product_sizes WHERE id=:sid", sid=sid)[0]

def holds(conn, lid, sid):
    return _q(conn, """
      SELECT p.id, COALESCE(mp.id,0), h.x, h.y
      FROM holes h
      JOIN placements p  ON p.hole_id=h.id AND p.layout_id=:lid AND p.set_id=:sid
      LEFT JOIN placements mp ON mp.hole_id=h.mirrored_hole_id
            AND mp.layout_id=:lid AND mp.set_id=:sid""", lid=lid, sid=sid)

def image(fname):
    r = requests.get(API.format(fname), timeout=15); r.raise_for_status()
    return Image.open(io.BytesIO(r.content))

def visualize_big_boards(min_climbs: int = 200_000, db_path: str = DB) -> None:
    """
    Show every board (layout_id, size_id) whose union-of-sets has
    ≥ `min_climbs` fully-contained climbs.  Each figure:
        • composite background (all sets),
        • red “id / index” label next to every visible hold.
    Close the window (or hit “next”) to advance to the next board.
    """
    big_boards = [s for s in board_stats(db_path) if s["climbs"] >= min_climbs]

    with sqlite3.connect(db_path) as conn:
        for st in big_boards:
            lid, sid, sets = st["layout_id"], st["size_id"], st["sets"]
            l, r, b, t     = edges(conn, sid)

            # ── composite background ────────────────────────────────────
            composite = None
            for set_id in sets:
                fname = next(
                    fn for s_id, s_set, fn in board_rows(conn, lid)
                    if s_id == sid and s_set == set_id
                )
                layer = image(fname).convert("RGBA")
                composite = layer if composite is None else Image.alpha_composite(composite, layer)

            # ── collect holds inside board edges ────────────────────────
            union = {}
            for set_id in sets:
                for pid, _mirrored, x, y in holds(conn, lid, set_id):   # ← note “_mirrored”
                    if l <= x <= r and b <= y <= t:
                        union[pid] = (x, y)


            ordered = sorted(union.items())          # deterministic order
            idx_map = {pid: i + 1 for i, (pid, _) in enumerate(ordered)}

            x_scale = composite.width  / (r - l)
            y_scale = composite.height / (t - b)

            # ── plot ─────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(6, 9))
            ax.imshow(composite)
            ax.axis("off")

            for pid, (x, y) in ordered:
                xp = (x - l) * x_scale
                yp = composite.height - (y - b) * y_scale
                ax.text(
                    xp,
                    yp,
                    f"{pid}\n\n{idx_map[pid]}",
                    color="red",
                    fontsize=6,
                    ha="center",
                    va="center",
                    linespacing=0.55,
                    linespacing=0.55,
                )

            ax.set_title(
                f"layout {lid}  •  size {sid}  •  sets {sets}  —  "
                f"{st['climbs']:,} climbs  •  {st['holds_total']} holds",
                fontsize=8,
            )
            plt.tight_layout()
            plt.show()

visualize_big_boards()