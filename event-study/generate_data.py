import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n_units = 1000
n_periods = 10
treatment_effect = 2.0  # Size of treatment effect
prob_treated = 0.6  # Proportion of treated units
missing_rate = 0  # Proportion of missing outcome data

# Create unit-time panel structure
units = np.arange(n_units)
times = np.arange(n_periods)
panel = pd.DataFrame([(i, t) for i in units for t in times], columns=["unit", "time"])

# -------- Simultaneous treatment --------
panel_simul = panel.copy()

# Assign treatment status (ever treated)
treated_units = np.random.choice(units, size=int(prob_treated * n_units), replace=False)
panel_simul["ever_treated"] = panel_simul["unit"].isin(treated_units).astype(int)
panel_simul["treatment_time"] = 5

# Treated only in post-treatment periods
panel_simul["treated"] = (
    panel_simul["ever_treated"] & (panel_simul["time"] >= panel_simul["treatment_time"])
).astype(int)

# Fixed effects
unit_fe = np.random.normal(0, 1, n_units)
time_fe = np.linspace(0, 1, n_periods)

panel_simul["unit_fe"] = panel_simul["unit"].map(lambda x: unit_fe[x])
panel_simul["time_fe"] = panel_simul["time"].map(lambda x: time_fe[x])
panel_simul["epsilon"] = np.random.normal(0, 1, panel_simul.shape[0])

# Outcome
panel_simul["y"] = (
    panel_simul["unit_fe"] +
    panel_simul["time_fe"] +
    panel_simul["treated"] * treatment_effect +
    panel_simul["epsilon"]
)

# Introduce missing data
mask_simul = np.random.rand(panel_simul.shape[0]) < missing_rate
panel_simul.loc[mask_simul, "y"] = np.nan

# Save
panel_simul = panel_simul.sort_values(by=["unit", "time"]).reset_index(drop=True)
panel_simul.to_csv("output/panel_data_simultaneous.csv", index=False)

# -------- Staggered treatment --------
panel_stagg = panel.copy()
panel_stagg["ever_treated"] = panel_stagg["unit"].isin(treated_units).astype(int)

# Randomly assign treatment time for treated units
treatment_times = {
    unit: np.random.choice(np.arange(3, 8))  # Staggered between periods 3 and 7
    for unit in treated_units
}

# Set treatment_time variable for all units
panel_stagg["treatment_time"] = panel_stagg["unit"].map(treatment_times).fillna(99999)

# Apply treatment based on staggered start time
panel_stagg["treated"] = (
    (panel_stagg["time"] >= panel_stagg["treatment_time"]) &
    (panel_stagg["ever_treated"] == 1)
).astype(int)

# Fixed effects
panel_stagg["unit_fe"] = panel_stagg["unit"].map(lambda x: unit_fe[x])
panel_stagg["time_fe"] = panel_stagg["time"].map(lambda x: time_fe[x])
panel_stagg["epsilon"] = np.random.normal(0, 1, panel_stagg.shape[0])

# Outcome
panel_stagg["y"] = (
    panel_stagg["unit_fe"] +
    panel_stagg["time_fe"] +
    panel_stagg["treated"] * treatment_effect +
    panel_stagg["epsilon"]
)

# Introduce missing data
mask_stagg = np.random.rand(panel_stagg.shape[0]) < missing_rate
panel_stagg.loc[mask_stagg, "y"] = np.nan

# Save
panel_stagg = panel_stagg.sort_values(by=["unit", "time"]).reset_index(drop=True)
panel_stagg.to_csv("output/panel_data_staggered.csv", index=False)

# Optional preview
print("Simultaneous Treatment Example:")
print(panel_simul.head())
print("\nStaggered Treatment Example:")
print(panel_stagg.head())
