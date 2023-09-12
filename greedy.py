import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

welder_df = pd.read_csv("welder_data.csv")
control_df = pd.read_csv("control_data.csv")

# distances between welders and controls
propensity_scores_welders = welder_df["Propensity_Score_Welder"].values.reshape(-1, 1)
propensity_scores_controls = control_df["Propensity_Score_Control"].values.reshape(-1, 1)
distance_matrix = euclidean_distances(propensity_scores_welders, propensity_scores_controls) ** 2

caliper = 0.086  # max diff between propensity scores
matched_pairs = []
matched_controls = set()  # Keep track of controls that have been matched

# Greedy Matching: find the best match for each welder
while len(matched_pairs) < len(welder_df):
    best_welder = None
    best_control = None
    best_distance = float('inf')

    #  each welder in the welder dataset
    for welder_idx in range(len(welder_df)):
        # Check if the welder has not been matched
        if welder_idx not in [pair[0] for pair in matched_pairs]:
            # Loop through each control in the control dataset
            for control_idx in range(len(control_df)):
                # Check if the control has not been matched
                if control_idx not in [pair[1] for pair in matched_pairs]:
                    distance = distance_matrix[welder_idx, control_idx] # distance
                    if distance <= caliper:

                        if distance < best_distance: #updates
                            best_distance = distance
                            best_welder = welder_idx
                            best_control = control_idx

    if best_welder is not None and best_control is not None:
        matched_pairs.append((best_welder, best_control))
        matched_controls.add(best_control)

for welder_idx, control_idx in matched_pairs:
    print(f"Welder {welder_idx + 1} matched with Control {control_idx + 1}")

print(welder_df)
print(control_df)
print(distance_matrix.shape)