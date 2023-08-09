import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

welder_data = {
    "ID": list(range(1, 22)),
    "Age_Welder": [38, 44, 39, 33, 35, 39, 27, 43, 39, 43, 41, 36, 35, 37, 39, 34, 35, 53, 38, 37, 38],
    "Race_Welder": ['C', 'C', 'C', 'AA', 'C', 'C', 'C', 'C', 'C', 'AA', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'],
    "Smoker_Welder": ['N', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'N', 'N', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N', 'Y'],
    "Propensity_Score_Welder": [0.46, 0.34, 0.57, 0.51, 0.65, 0.57, 0.68, 0.49, 0.57, 0.20, 0.53, 0.50, 0.52, 0.48, 0.57, 0.54, 0.65, 0.19, 0.60, 0.48, 0.60]
}

control_data = {
    "ID": list(range(1, 28)),
    "Age_Control": [45, 47, 39, 41, 34, 31, 35, 41, 34, 50, 44, 42, 40, 44, 35, 38, 36, 52, 36, 42, 38, 30, 38, 40, 38, 42, 38, 40],
    "Race_Control": ['C', 'C', 'C', 'C', 'C', 'AA', 'C', 'AA', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'],
    "Smoker_Control": ['N', 'N', 'Y', 'N', 'N', 'Y', 'N', 'Y', 'N', 'N', 'Y', 'N', 'N', 'N', 'N', 'Y', 'N', 'N', 'Y', 'N', 'N', 'N', 'Y', 'N', 'Y', 'N', 'Y', 'N'],
    "Propensity_Score_Control": [0.32, 0.28, 0.57, 0.40, 0.67, 0.55, 0.65, 0.35, 0.54, 0.23, 0.47, 0.38, 0.42, 0.34, 0.52, 0.46, 0.64, 0.20, 0.64, 0.38, 0.63, 0.46, 0.64, 0.38, 0.60, 0.38, 0.46, 0.46]
}

welder_df = pd.DataFrame(welder_data)
control_df = pd.DataFrame(control_data)

# distances between welders and controls. mentioned somwehre in book
propensity_scores_welders = welder_df["Propensity_Score_Welder"].values.reshape(-1, 1)
propensity_scores_controls = control_df["Propensity_Score_Control"].values.reshape(-1, 1)
distance_matrix = euclidean_distances(propensity_scores_welders, propensity_scores_controls) ** 2

caliper = 0.086  # max diff between propensity scores
matched_pairs = []
matched_controls = []

# matching: find controls within range
# and select the control with the closest propensity score

for welder_idx in range(len(distance_matrix)):
    potential_controls = np.where(distance_matrix[welder_idx] <= caliper)[0]
    if len(potential_controls) > 0:
        best_control = potential_controls[np.argmin(distance_matrix[welder_idx, potential_controls])]
        matched_pairs.append((welder_idx, best_control))
        matched_controls.append(best_control)

for welder_idx, control_idx in matched_pairs:
    print(f"Welder {welder_idx + 1} matched with Control {control_idx + 1}")
