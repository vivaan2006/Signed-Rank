import numpy as np
import pandas as pd
from psmpy import PsmPy

welder_df = pd.read_csv("welder_data.csv")
control_df = pd.read_csv("control_data.csv")

propensity_scores_welders = welder_df["Propensity_Score_Welder"].values
propensity_scores_controls = control_df["Propensity_Score_Control"].values

caliper = 0.086
# chapter 8 i tink
propensity_matrix = np.column_stack((propensity_scores_welders, propensity_scores_controls))

# match but i think this causes an error
matched_pairs = PsmPy.PsmPy(propensity_matrix, caliper=caliper)

# index of matched pairs
welder_indices = matched_pairs[:, 0].astype(int)
control_indices = matched_pairs[:, 1].astype(int)

for i, (welder_idx, control_idx) in enumerate(zip(welder_indices, control_indices)):
    print(f"Welder {welder_idx + 1} matched with Control {control_idx + 1}")

print(welder_df)
print(control_df)
