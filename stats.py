#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

#%%
# Load the training data
try:
    train_df = pd.read_csv("climbs_train.csv")
except FileNotFoundError:
    print("climbs_train.csv not found. Please run main.py first to generate it.")
    exit()

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
