"Draw climb"

import sqlite3, io, requests, matplotlib.pyplot as plt
from PIL import Image
from typing import List, Union, Optional

DB  = "kilter_dataset.csv"
API = "https://api.kilterboardapp.com/img/{}"

# ── tiny helpers ────────────────────────────────────────────────────────────
_q = lambda c, sql, **p: c.execute(sql, p).fetchall()

def image(fname: str) -> Image.Image:
    r = requests.get(API.format(fname), timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")

def parse_frames(frames: str) -> dict[int, int]:
    return {int(p): int(c) for p, c in
            (seg.split("r") for seg in frames.split("p") if seg)}

def edges(conn: sqlite3.Connection, sid: int):
    return _q(conn, "SELECT edge_left,edge_right,edge_bottom,edge_top "
                    "FROM product_sizes WHERE id=:sid", sid=sid)[0]

def holds(conn: sqlite3.Connection, lid: int, sid: int):
    return _q(conn, """
      SELECT p.id, COALESCE(mp.id,0), h.x, h.y
      FROM holes h
      JOIN placements p  ON p.hole_id=h.id AND p.layout_id=:lid AND p.set_id=:sid
      LEFT JOIN placements mp ON mp.hole_id=h.mirrored_hole_id
            AND mp.layout_id=:lid AND mp.set_id=:sid""", lid=lid, sid=sid)

def color_map(conn: sqlite3.Connection, lid: int):
    return dict(_q(conn, """
      SELECT placement_roles.id,'#'||placement_roles.screen_color
      FROM placement_roles JOIN layouts ON layouts.product_id=placement_roles.product_id
      WHERE layouts.id=:lid""", lid=lid))

# ── main drawing function ──────────────────────────────────────────────────
def draw_climb(
    climb_uuid: str,
    layout_id: int,
    product_size_id: int,
    set_ids: List[int],
    *,
    title: Optional[str] = None,
    db_path: str = DB,
) -> None:
    """
    Display the climb `climb_uuid` on the board defined by (layout_id,
    product_size_id), overlaying *all* `set_ids`.

    Parameters
    ----------
    climb_uuid       : str     – UUID of the climb
    layout_id        : int     – board layout ID
    product_size_id  : int     – product size ID
    set_ids          : List[int] – all sets whose layers form the board
    title            : str | None – figure title; defaults to climb name
    db_path          : str     – path to the SQLite DB
    """
    with sqlite3.connect(db_path) as conn:
        # ── fetch climb info, colors, frames ──────────────────────────────
        row = conn.execute(
            "SELECT name,frames FROM climbs WHERE uuid=?", (climb_uuid,)
        ).fetchone()
        if row is None:
            raise ValueError(f"climb uuid “{climb_uuid}” not found")
        climb_name, frames = row
        fmap = parse_frames(frames)
        cmap = color_map(conn, layout_id)

        # ── composite background image ────────────────────────────────────
        composite: Union[Image.Image, None] = None
        for sid in set_ids:
            (fname,) = conn.execute(
                """SELECT image_filename
                   FROM product_sizes_layouts_sets
                   WHERE layout_id=? AND product_size_id=? AND set_id=?""",
                (layout_id, product_size_id, sid),
            ).fetchone()
            layer = image(fname)
            composite = layer if composite is None else Image.alpha_composite(composite, layer)

        # ── board geometry & scaling ──────────────────────────────────────
        l, r, b, t = edges(conn, product_size_id)
        xs, ys = composite.width / (r - l), composite.height / (t - b)

        # ── plot ──────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 9))
        ax.imshow(composite)
        ax.axis("off")

        for sid in set_ids:
            for pid, _mirrored, x, y in holds(conn, layout_id, sid):
                if pid not in fmap:
                    continue
                cx, cy = (x - l) * xs, composite.height - (y - b) * ys
                ax.scatter(
                    cx,
                    cy,
                    s=300,
                    facecolors="none",
                    edgecolors=cmap.get(fmap[pid], "#ff0000"),
                    linewidths=3,
                )

        ax.set_title(title or climb_name, fontsize=10)
        plt.tight_layout()
        plt.show()
        return fig, ax