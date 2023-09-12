#same as before but i just shuffeled order
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

welder_df = pd.read_csv("welder_data.csv")
control_df = pd.read_csv("control_data.csv")

propensity_scores_welders = welder_df["Propensity_Score_Welder"].values.reshape(-1, 1)
propensity_scores_controls = control_df["Propensity_Score_Control"].values.reshape(-1, 1)

# Distance matrix between welders and controls
distance_matrix = euclidean_distances(propensity_scores_welders, propensity_scores_controls) ** 2

caliper = 0.086

# random shuffle welders
welder_indices = np.arange(len(distance_matrix))
np.random.shuffle(welder_indices)

# Store and track matched pairs
matched_pairs = []
matched_controls = set()

# Matching find controls within range and getting one w closest prop score
for welder_idx in welder_indices:
    potential_controls = np.where(distance_matrix[welder_idx] <= caliper)[0]
    if len(potential_controls) > 0:
        # Exclude matched controls to stop reincluding controls
        potential_controls = [c for c in potential_controls if c not in matched_controls]
        if len(potential_controls) > 0:
            best_control = potential_controls[np.argmin(distance_matrix[welder_idx, potential_controls])]
            matched_pairs.append((welder_idx, best_control))
            matched_controls.add(best_control)

for welder_idx, control_idx in matched_pairs:
    print(f"Welder {welder_idx + 1} matched with Control {control_idx + 1}")

print(welder_df)
print(control_df)

print(distance_matrix.shape)
