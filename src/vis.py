from PIL import Image
import matplotlib.pyplot as plt

def draw_climb(triples, title=None, with_order=False):
    img = Image.open("board.png")
    l, r, b, t = 0, 144, 0, 156
    xs, ys = img.width / (r - l), img.height / (t - b)
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.imshow(img)
    ax.axis("off")
    role_styles = {
        "s": {"edgecolors": "#00DD00", "facecolors": "none", "linewidths": 3, "s": 350},   # start
        "m": {"edgecolors": "#00FFFF", "facecolors": "none", "linewidths": 3, "s": 300},   # middle
        "e": {"edgecolors": "#FF00FF", "facecolors": "none", "linewidths": 3, "s": 350},   # end
        "f": {"edgecolors": "#FFA500", "facecolors": "none", "linewidths": 3, "s": 200},   # foot
        None: {"edgecolors": "black", "facecolors": "none", "linewidths": 2, "s": 150},
    }
    for idx, (x, y, role) in enumerate(triples, 1):
        cx, cy = (x - l) * xs, img.height - (y - b) * ys
        style = role_styles.get(role, role_styles[None])
        ax.scatter(cx, cy, **style)
        if with_order:
            ax.text(cx - 35, cy, str(idx), color="red", fontsize=10, ha="center", va="center", fontweight="bold")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

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
# draw_climb(holds, title="run over me, 7a/V6")

# val_df = pd.read_csv("climbs_val.csv")
# row = val_df.iloc[3]
# holds = row["holds_xy"]
# holds = eval(holds) if isinstance(holds, str) else holds
# name = row["name"]
# draw_climb(holds, title=name, with_order=True)