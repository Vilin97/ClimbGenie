from PIL import Image
import matplotlib.pyplot as plt

def draw_climb(xy_pairs, roles, title=None, annotations=None):
    img = Image.open("board.png")
    l, r, b, t = 0, 144, 0, 156
    xs, ys = img.width / (r - l), img.height / (t - b)
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.imshow(img)
    ax.axis("off")
    linewidth = 2.5
    role_styles = {
        "s": {"edgecolors": "#00DD00", "facecolors": "none", "linewidths": linewidth, "s": 300},   # start
        "m": {"edgecolors": "#00FFFF", "facecolors": "none", "linewidths": linewidth, "s": 300},   # middle
        "e": {"edgecolors": "#FF00FF", "facecolors": "none", "linewidths": linewidth, "s": 300},   # end
        "f": {"edgecolors": "#FFA500", "facecolors": "none", "linewidths": linewidth, "s": 200},   # foot
        None: {"edgecolors": "black", "facecolors": "none", "linewidths": 2, "s": 150},
    }
    triples = [(x, y, role) for (x, y), role in zip(xy_pairs, roles)]
    for idx, (x, y, role) in enumerate(triples):
        cx, cy = (x - l) * xs, img.height - (y - b) * ys
        style = role_styles.get(role, role_styles[None])
        ax.scatter(cx, cy, **style)
        if annotations:
            ax.text(cx, cy-25, annotations[idx], color="red", fontsize=8, ha="center", va="center", fontweight="bold")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def make_title(row):
    return f"{row['name']} | {row['grade']} @ {int(row['angle'])}° | {int(row['ascents'])} ascents"

def draw_climb_row(row):
    holds_xy = eval(row["holds_xy"])
    roles = row["roles"]
    holds_indices = eval(row["holds_indices"])
    draw_climb(holds_xy, roles, title=make_title(row), annotations=holds_indices)

# Examples:
# holds = [(32, 8, 'f'),
#  (56, 48, 's'),
#  (40, 56, 's'),
#  (112, 64, 'f'),
#  (72, 80, 'm'),
#  (96, 104, 'm'),
#  (96, 128, 'm'),
#  (88, 152, 'e'),
#  (60, 4, 'f')]
# draw_climb(holds, title="run over me, 7a/V6 @ 30°")