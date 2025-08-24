#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import ast
import re
import numpy as np

#%%
# Load the training data
train_df = pd.read_csv("climbs_train.csv")

#%%
# Compute and print the correlation
correlation = train_df['angle'].corr(train_df['difficulty'])
print(f"Correlation between angle and difficulty: {correlation:.4f}")

# Create the scatter plot with a line of best fit
plt.figure(figsize=(12, 8))
sns.regplot(data=train_df, x='angle', y='difficulty', scatter_kws={'alpha': 0.5, 's': 10})

# Add titles and labels for clarity
plt.title(f'Climb Difficulty vs. Angle (Correlation: {correlation:.2f})')
plt.xlabel('Angle (degrees)')
plt.ylabel('Difficulty (numeric)')
plt.grid(True)
plt.show()

#%%
# find climbs that have lower difficulty at higher angles

# Group by climb uuid
grouped = train_df.groupby('uuid')

# List to store the interesting pairs
interesting_pairs = []

# Iterate over each group of climbs with the same uuid
for _, group in grouped:
    # We only care about climbs that appear more than once (at different angles)
    if len(group) > 1:
        # Iterate through all combinations of 2 settings for the same climb
        for climb1, climb2 in combinations(group.to_dict('records'), 2):
            # Check for the condition: one has steeper angle but lower difficulty
            c1_steeper_easier = climb1['angle'] > climb2['angle'] and climb1['difficulty'] < climb2['difficulty']
            c2_steeper_easier = climb2['angle'] > climb1['angle'] and climb2['difficulty'] < climb1['difficulty']

            if c1_steeper_easier:
                # Store as (less_steep, more_steep_but_easier)
                interesting_pairs.append((climb2, climb1))
            elif c2_steeper_easier:
                # Store as (less_steep, more_steep_but_easier)
                interesting_pairs.append((climb1, climb2))

num_climbs_with_inconsistent_grading = len(set([x[0]['uuid'] for x in interesting_pairs]))
print(f"Number of climbs with inconsistent grading (steeper angle, lower difficulty): {num_climbs_with_inconsistent_grading}")

#%%
# Print the results
print(f"Found {len(interesting_pairs)} pairs of climbs where a steeper angle has a lower difficulty:")
for i, (climb_A, climb_B) in enumerate(interesting_pairs[:10]):
    print(f"\n--- Pair {i+1}: {climb_A['name']}  ---")
    print(f"  Setting 1: Angle={climb_A['angle']}°, Difficulty={climb_A['difficulty']:.2f}, Grade='{climb_A['grade']}', Ascents={climb_A['ascents']}")
    print(f"  Setting 2: Angle={climb_B['angle']}°, Difficulty={climb_B['difficulty']:.2f}, Grade='{climb_B['grade']}', Ascents={climb_B['ascents']}")

#%%
# Compute number of holds for each climb
train_df['num_holds'] = train_df['holds_xy'].apply(lambda x: len(ast.literal_eval(x)))

plt.figure(figsize=(10, 6))
sns.histplot(train_df['num_holds'], bins=20, kde=False, color='skyblue')
plt.title('Histogram of Number of Holds per Climb')
plt.xlabel('Number of Holds')
plt.ylabel('Count')
plt.grid(True)
plt.show()

#%%

fig, ax = plt.subplots(figsize=(10, 6))
data = train_df['grade_us'].dropna().astype(int)
total = len(data)

sns.histplot(data, bins=range(0, 15), color='skyblue', discrete=True, ax=ax)

ax.set_title('Histogram of V grade (V0 – V13)')
ax.set_xlabel('V grade')
ax.set_ylabel('Count')
ax.set_xticks(range(0, 14))
ax.set_xlim(-0.5, 13.5)
ax.grid(True)

# Annotate each bar with percentage in small red text
patches = ax.patches[:-1]
if total > 0:
    heights = [p.get_height() for p in patches]
    maxh = max(heights) if heights else 0
    y_offset = maxh * 0.02 if maxh > 0 else 0.1
    for p in patches:
        h = p.get_height()
        if h <= 0:
            continue
        x = p.get_x() + p.get_width() / 2
        pct = h / total * 100
        ax.text(x, h + y_offset, f"{pct:.1f}%", color='red', fontsize=8, ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Histogram of difficulty (numeric) with percentage annotations
fig, ax = plt.subplots(figsize=(10, 6))
data = train_df['difficulty'].dropna()
total = len(data)

# compute bin edges and centers so we can place a tick for each bin
n_bins = 21
counts, bin_edges = np.histogram(data, bins=n_bins)
centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

sns.histplot(data, bins=bin_edges, kde=False, color='skyblue', ax=ax)

ax.set_title('Histogram of Difficulty')
ax.set_xlabel('Difficulty')
ax.set_ylabel('Count')
ax.grid(True)

# set a tick at each bin center
ax.set_xticks(centers)
ax.set_xticklabels([f"{int(c-0.5)}" for c in centers], rotation=45, ha='right', fontsize=8)

patches = ax.patches
if total > 0 and patches:
    heights = [p.get_height() for p in patches]
    maxh = max(heights) if heights else 0
    y_offset = maxh * 0.02 if maxh > 0 else 0.1
    for p in patches:
        h = p.get_height()
        if h <= 0:
            continue
        x = p.get_x() + p.get_width() / 2
        pct = h / total * 100
        ax.text(x, h + y_offset, f"{pct:.1f}%", color='red', fontsize=8, ha='center', va='bottom')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
data = train_df['angle'].dropna()
total = len(data)

# compute bin edges and centers for angle histogram
n_bins = 14
counts, bin_edges = np.histogram(data, bins=n_bins)
centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

sns.histplot(data, bins=bin_edges, kde=False, color='skyblue', ax=ax)

ax.set_title('Histogram of Angle')
ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('Count')
ax.grid(True)

# set a tick at each bin center (rounded to 1 decimal or integer as appropriate)
ax.set_xticks(centers)
ax.set_xticklabels([f"{int(c-2.5)}" for c in centers], rotation=45, ha='right', fontsize=8)

patches = ax.patches
if total > 0 and patches:
    heights = [p.get_height() for p in patches]
    maxh = max(heights) if heights else 0
    y_offset = maxh * 0.02 if maxh > 0 else 0.1
    for p in patches:
        h = p.get_height()
        if h <= 0:
            continue
        x = p.get_x() + p.get_width() / 2
        pct = h / total * 100
        ax.text(x, h + y_offset, f"{pct:.1f}%", color='red', fontsize=8, ha='center', va='bottom')

plt.tight_layout()
plt.show()