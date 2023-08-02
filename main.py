import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.stats import wilcoxon

data = pd.read_csv('/Users/vivaan/PycharmProjects/Observations/random experiment - Sheet1.csv')

unique_help_values = data['Help'].unique()
if len(unique_help_values) > 2 or not all(value in [0, 1] for value in unique_help_values):
    raise ValueError("The 'Help' column must contain only 0s and 1s for logistic regression.")

X = data[['GPA']]
X = sm.add_constant(X)
y = data['Help']

logit_model = sm.Logit(y, X)
logit_result = logit_model.fit()


propensity_scores = logit_result.predict(X)

threshold = 0.1
matched_data = data.copy()
matched_data['Propensity_Score'] = propensity_scores

gpa_group_1 = matched_data[matched_data['Help'] == 1]
gpa_group_0 = matched_data[matched_data['Help'] == 0]

def find_closest_match(row):
    group_0_matches = gpa_group_0[
        np.abs(gpa_group_0['Propensity_Score'] - row['Propensity_Score']) <= threshold
    ]
    return group_0_matches.iloc[0] if not group_0_matches.empty else None

matches = gpa_group_1.apply(find_closest_match, axis=1)

# Drop the unused values
matched_data = matched_data.dropna(subset=['Help'])
matched_data = matched_data.dropna(subset=['Propensity_Score'])

matched_data = pd.concat([gpa_group_1, matches], axis=0)

# Separate GPA for the groups
gpa_group_1 = matched_data[matched_data['Help'] == 1]['GPA']
gpa_group_0 = matched_data[matched_data['Help'] == 0]['GPA']


statistic, p_value = wilcoxon(gpa_group_1, gpa_group_0)
O
print(f"Wilcoxon Signed-Rank Test:")
print(f"Statistic: {statistic}")
print(f"P-value: {p_value}")
print("")

# wasnt sure what p val to use
if p_value < 0.05:
    print("difference is significant.")
else:
    print("no difference")
