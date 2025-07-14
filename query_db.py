#%%
""" Dispplay one kilter board image """
import sqlite3
import pandas as pd
import requests
from PIL import Image
import matplotlib.pyplot as plt
import io

def query_product_sizes_layouts_sets():
    """Query the product_sizes_layouts_sets table and return as pandas DataFrame"""
    
    # Connect to the SQLite database
    conn = sqlite3.connect('kilter_dataset.csv')
    
    try:
        # Query the table
        query = "SELECT * FROM product_sizes_layouts_sets"
        df = pd.read_sql_query(query, conn)
        
        return df
    
    finally:
        # Close the connection
        conn.close()

def display_image(filename):
    """Download and display a single image from Kilter Board API"""
    url = f"https://api.kilterboardapp.com/img/{filename}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image: {filename}")
        plt.show()
        
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

# Execute the query and get DataFrame
df = query_product_sizes_layouts_sets()
print(df.head())
display_image("product_sizes_layouts_sets/47.png")
#%%
""" Make a PDF with all kilter board images """
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Spacer, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import tempfile
import os
from tqdm import tqdm

def create_pdf_from_images():
    """Download all images and concatenate them into a PDF"""
    pdf_filename = "kilter_board_images.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4)
    story = []
    temp_files = []  # Keep track of temp files to clean up later
    styles = getSampleStyleSheet()
    
    # Get page dimensions
    page_width, page_height = A4
    max_width = page_width - 2*inch  # Leave margins
    max_height = page_height - 3*inch  # Leave space for title and margins
    
    for index, row in tqdm(list(df.iterrows())):
        filename = row["image_filename"]
        url = f"https://api.kilterboardapp.com/img/{filename}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Create temporary file for the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            temp_files.append(temp_path)  # Track temp file for later cleanup
            
            # Add filename as title
            title = Paragraph(f"Image: {filename}", styles['Heading2'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Get original image dimensions
            img_pil = Image.open(temp_path)
            original_width, original_height = img_pil.size
            
            # Convert pixels to inches (assuming 72 DPI)
            width_inches = original_width / 72.0
            height_inches = original_height / 72.0
            
            # Scale to fit page while maintaining aspect ratio
            scale_factor = min(max_width / (width_inches * inch), max_height / (height_inches * inch))
            if scale_factor > 1:
                scale_factor = 1  # Don't upscale
            
            final_width = width_inches * inch * scale_factor
            final_height = height_inches * inch * scale_factor
            
            # Add image to PDF with scaled size
            img = RLImage(temp_path, width=final_width, height=final_height)
            story.append(img)
            
            # Add page break after each image (except the last one)
            if index < len(df) - 1:
                story.append(PageBreak())
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Build PDF first, then clean up temp files
    doc.build(story)
    
    # Clean up temporary files
    for temp_path in temp_files:
        try:
            os.unlink(temp_path)
        except OSError:
            pass  # File already deleted or doesn't exist
    
    print(f"PDF created: {pdf_filename}")

create_pdf_from_images()

#%%
"""
Generate a multi-page PDF that shows every board image on which the given climb
appears, with its holds circled.

INPUT  : any climb name (case-insensitive, punctuation ignored)
OUTPUT : <normalized-name>.pdf in the current directory
"""

import sqlite3, io, requests, re, os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sqlite3, re
from typing import List, Tuple, Dict, Set

DB   = "kilter_dataset.csv"                   # ← SQLite file
NAME = "swooped"                     # ← put your climb name here
API  = "https://api.kilterboardapp.com/img/{}"

# ── tiny helper ─────────────────────────────────────────────────────────────
_q = lambda c, sql, **p: c.execute(sql, p).fetchall()
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())

# ── lookup climb by (fuzzy) name ────────────────────────────────────────────
def climb_row(conn, raw_name):
    n = norm(raw_name)
    for uuid, name, layout_id, frames in _q(conn,
        "SELECT uuid,name,layout_id,frames FROM climbs"):
        if norm(name) == n:
            return uuid, name, layout_id, frames
    raise ValueError("climb not found")

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

def color_map(conn, lid):
    return dict(_q(conn, """
      SELECT placement_roles.id,'#'||placement_roles.screen_color
      FROM placement_roles JOIN layouts ON layouts.product_id=placement_roles.product_id
      WHERE layouts.id=:lid""", lid=lid))

# ── other helpers ───────────────────────────────────────────────────────────
def parse_frames(frames):
    return {int(p):int(c) for p,c in
            (seg.split("r") for seg in frames.split("p") if seg)}

def image(fname):
    r = requests.get(API.format(fname), timeout=15); r.raise_for_status()
    return Image.open(io.BytesIO(r.content))

def plot(ax, img, holds, fm, l,r,b,t, cmap):
    ax.imshow(img); ax.axis("off")
    xs,ys = img.width/(r-l), img.height/(t-b)
    for pid,_,x,y in holds:
        if pid not in fm: continue
        cx,cy = (x-l)*xs, img.height-(y-b)*ys
        ax.scatter(cx,cy,s=300,facecolors="none",
                   edgecolors=cmap.get(fm[pid],"#ff0000"),linewidths=3)

def board_variants_for_climb(
        climb_name: str,
        db_path: str = "kilter_dataset.csv",
        complete_only: bool = True
) -> List[Tuple[int, int, int]]:
    """
    Return every (layout_id, product_size_id, set_id) triple whose board
    contains all (or at least one, if complete_only=False) of the holds that
    appear in `frames` for the climb named `climb_name`.

    The check is geometric: a hold counts only if its (x,y) lies strictly
    inside [edge_left, edge_right] × [edge_bottom, edge_top] for that size.

    Parameters
    ----------
    climb_name    : str   – name of the climb (case-insensitive, punctuation ignored)
    db_path       : str   – path to the SQLite DB
    complete_only : bool  –   True  → require *all* holds present (default)
                              False → at least one hold present

    Returns
    -------
    List[(layout_id, size_id, set_id)]
    """
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())

    with sqlite3.connect(db_path) as conn:
        # ── locate climb ───────────────────────────────────────────────────
        climb_row = conn.execute(
            "SELECT uuid,name,layout_id,frames FROM climbs"
        ).fetchall()
        try:
            uuid, name, layout_id, frames = next(
                row for row in climb_row if norm(row[1]) == norm(climb_name)
            )
        except StopIteration:
            raise ValueError(f"climb “{climb_name}” not found")

        frame_ids: Set[int] = {int(seg.split("r")[0])
                               for seg in frames.split("p") if seg}

        # ── helper: ids that are inside edges for (size_id,set_id) ─────────
        def in_bounds_ids(size_id: int, set_id: int) -> Set[int]:
            l, r, b, t = conn.execute("""
                SELECT edge_left,edge_right,edge_bottom,edge_top
                FROM product_sizes WHERE id=?""", (size_id,)).fetchone()

            rows = conn.execute("""
                SELECT p.id
                FROM placements p
                JOIN holes h ON h.id = p.hole_id
                WHERE p.layout_id=? AND p.set_id=?
                  AND h.x>? AND h.x<? AND h.y>? AND h.y<?""",
                (layout_id, set_id, l, r, b, t)).fetchall()
            return {pid for (pid,) in rows}

        # ── iterate over all board combos for this layout ─────────────────
        combos = conn.execute("""
            SELECT product_size_id,set_id
            FROM product_sizes_layouts_sets
            WHERE layout_id=?""", (layout_id,)).fetchall()

        valid = []
        for size_id, set_id in combos:
            inside = in_bounds_ids(size_id, set_id)
            present = frame_ids & inside
            if (complete_only and len(present) == len(frame_ids)) or (
                not complete_only and present):
                valid.append((layout_id, size_id, set_id))

    # sort (size_id, set_id) for readability
    return sorted(valid, key=lambda x: (x[1], x[2]))

# ── main ────────────────────────────────────────────────────────────────────
def pdf_for_name(name: str, db_path: str = DB) -> None:
    """
    Build a PDF that shows the climb only on boards which contain *all* of its holds.
    """
    with sqlite3.connect(db_path) as conn:
        uuid, realname, lid, frames = climb_row(conn, name)
        fm   = parse_frames(frames)
        cmap = color_map(conn, lid)

        # — boards that contain every hold —
        boards = board_variants_for_climb(realname, db_path, complete_only=True)

        pdf_name = f"{norm(realname)}.pdf"
        with PdfPages(pdf_name) as pdf:
            for _lid, size_id, set_id in boards:
                # image filename for this exact size / set
                (fname,) = conn.execute(
                    """SELECT image_filename
                       FROM product_sizes_layouts_sets
                       WHERE layout_id=? AND product_size_id=? AND set_id=?""",
                    (lid, size_id, set_id)
                ).fetchone()

                try:
                    img = image(fname)
                except Exception as e:
                    print("skip", fname, e)
                    continue

                l, r, b, t = edges(conn, size_id)
                fig, ax = plt.subplots(figsize=(6, 9))
                ax.set_title(f"{realname}  •  size {size_id}  set {set_id}", fontsize=9)
                plot(ax, img, holds(conn, lid, set_id), fm, l, r, b, t, cmap)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print("PDF saved:", os.path.abspath(pdf_name))

# ── run ─────────────────────────────────────────────────────────────────────
pdf_for_name(NAME)