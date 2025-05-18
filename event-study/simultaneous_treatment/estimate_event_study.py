import pandas as pd
import numpy as np
import pyfixest as pf

panel = pd.read_csv("output/panel_data.csv")
panel.head()

# Estimate static DID
# CRV1 matches R's fixest standard error
static_did = pf.feols(
    fml="y ~ treated|unit + time ", 
    data=panel, 
    vcov={"CRV1": "unit_fe"}
)
pf.etable(static_did)

# Estimate dynamic DID
# Compare three methods: have never treated units get -1 event time, 0 event time, and 999 event time (and omit that 999 indicator). All methods omit the -1 indicator.
# Note, you can skip the indicator creation by writing a formula like "y ~ i(event_time, ref = -1) | unit + time", but that results in really messy coefficient names like C(event_time, contr.treatment(base=-1))[T.-5]

def run_dynamic_did(panel, treatment_time, control_event_time_value, drop_terms):
    df = panel.copy()

    # Assign event_time
    df["event_time"] = np.where(
        df["ever_treated"] == 1,
        df["time"] - treatment_time,
        control_event_time_value
    )

    # Create dummies and rename
    leads_lags = pd.get_dummies(df["event_time"], prefix="T").astype(int)
    leads_lags.columns = [col.replace("-", "minus_") for col in leads_lags.columns]

    # Drop reference + optional control group dummy
    leads_lags = leads_lags.drop(columns=drop_terms, errors="ignore")

    # Add to data and estimate
    df = pd.concat([df, leads_lags], axis=1)
    rhs_terms = " + ".join(leads_lags.columns)
    formula = f"y ~ {rhs_terms} | unit + time"

    return pf.feols(formula, data=df, vcov={"CRV1": "unit_fe"})

treatment_time = 5

dynamic_did_minus_1 = run_dynamic_did(
    panel, treatment_time, control_event_time_value=-1, drop_terms=["T_minus_1"]
)

dynamic_did_0 = run_dynamic_did(
    panel, treatment_time, control_event_time_value=0, drop_terms=["T_minus_1"]
)

dynamic_did_999 = run_dynamic_did(
    panel, treatment_time, control_event_time_value=999, drop_terms=["T_minus_1", "T_999"]
)

pf.etable([dynamic_did_minus_1, dynamic_did_0, dynamic_did_999])

# Using linearmodels doesn't match R's fixest + is clunkier
# from linearmodels.panel import PanelOLS
# panel = panel.set_index(['unit', 'time'])

# static_did_lm = PanelOLS.from_formula("y ~ treated + EntityEffects + TimeEffects", data=panel)
# static_did_lm = static_did_lm.fit(cov_type='clustered', cluster_entity=True)

# print(static_did_lm.summary)