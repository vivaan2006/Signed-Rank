import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# Load necessary R libraries
tidyverse = importr('tidyverse')
MatchIt = importr('MatchIt')

# Activate pandas conversion for R objects
pandas2ri.activate()

# Read CSV files into R data frames
ro.r.assign("welder_df", pd.read_csv("welder_data.csv"))
ro.r.assign("control_df", pd.read_csv("control_data.csv"))

# Propensity score data
propensity_scores_welders = ro.r("welder_df$Propensity_Score_Welder")
propensity_scores_controls = ro.r("control_df$Propensity_Score_Control")

# Create a data frame with propensity scores
propensity_df = pd.DataFrame({
    "Welder": propensity_scores_welders,
    "Control": propensity_scores_controls
})

# Calculate the Euclidean distance matrix
ro.r.assign("propensity_df", propensity_df)
distance_matrix_r = ro.r("as.matrix(dist(propensity_df))")

# Define the caliper value
caliper = 0.086

# Perform propensity score matching using MatchIt
matched_pairs_r = MatchIt.matchit(
    formula="Welder ~ Control",
    data=propensity_df,
    method="exact",
    caliper=caliper
)

# Extract the matched pairs
matched_pairs = list(matched_pairs_r.rx2("match.matrix"))

# Print the matched pairs
for pair in matched_pairs:
    welder_idx = pair[0] + 1  # R uses 0-based indexing
    control_idx = pair[1] + 1  # R uses 0-based indexing
    print(f"Welder {welder_idx} matched with Control {control_idx}")

# Print the original data frames and the distance matrix
print(welder_df)
print(control_df)
print(distance_matrix_r.shape)