import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n_units = 100
n_periods = 10
treatment_time = 5  # Time period when treatment starts
treatment_effect = 2.0  # Size of treatment effect
prob_treated = 0.6  # Proportion of treated units
missing_rate = 0.1  # Proportion of missing outcome data

# Create unit-time panel structure
units = np.arange(n_units)
times = np.arange(n_periods)
panel = pd.DataFrame([(i, t) for i in units for t in times], columns=['unit', 'time'])

# Assign treatment status (ever treated)
treated_units = np.random.choice(units, size=int(prob_treated * n_units), replace=False)
panel['treated'] = panel['unit'].isin(treated_units)
panel['ever_treated'] = panel['treated'].astype(int)

# Indicator for post-treatment period
panel['post_treatment'] = (panel['time'] >= treatment_time).astype(int)

# Actual treatment occurs only for treated units in post-treatment periods
panel['treatment'] = panel['treated'] & (panel['time'] >= treatment_time)

# Unit fixed effects and time trends
unit_fe = np.random.normal(0, 1, n_units)
time_fe = np.linspace(0, 1, n_periods)

panel['unit_fe'] = panel['unit'].map(lambda x: unit_fe[x])
panel['time_fe'] = panel['time'].map(lambda x: time_fe[x])

# Random noise
panel['epsilon'] = np.random.normal(0, 1, panel.shape[0])

# Generate outcome variable
panel['y'] = (
    panel['unit_fe'] +
    panel['time_fe'] +
    panel['treatment'] * treatment_effect +
    panel['epsilon']
)

# Introduce missing data randomly
mask = np.random.rand(panel.shape[0]) < missing_rate
panel.loc[mask, 'y'] = np.nan

# Optional: sort and reset index
panel = panel.sort_values(by=['unit', 'time']).reset_index(drop=True)

# Preview data
print(panel.head())