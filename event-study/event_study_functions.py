import pandas as pd
import numpy as np
import pyfixest as pf
import re

def prep_dynamic_did_data(
    panel_path, 
    control_event_time_value=0, 
    drop_terms=["T_minus_1"]
):
    df = pd.read_csv(panel_path)

    # Assign event_time
    # The if-else statement bakes in the interaction
    df["event_time"] = np.where(
        df["ever_treated"] == 1,
        df["time"] - df["treatment_time"],
        control_event_time_value
    )

    # Create dummies and rename
    # Note: writing a formula like "y ~ i(event_time, ref = -1) | unit + time", skips this step, but that leads to messy coefficient names
    leads_lags = pd.get_dummies(df["event_time"].astype(int), prefix="T")
    leads_lags.columns = [col.replace("-", "minus_") for col in leads_lags.columns]

    # Drop reference + optional control group dummy
    leads_lags = leads_lags.drop(columns=drop_terms, errors="ignore")

    # Add to data
    df = pd.concat([df, leads_lags], axis=1)
    return df

def run_dynamic_did(
    panel_path, 
    control_event_time_value=0, 
    drop_terms=["T_minus_1"]
):
    df = prep_dynamic_did_data(panel_path, control_event_time_value, drop_terms)

    # Construct formula
    rhs_terms = " + ".join(df.filter(regex="^T_").columns)
    formula = f"y ~ {rhs_terms} | unit + time"

    return pf.feols(formula, data=df, vcov={"CRV1": "unit_fe"})


def plot_event_study(model, ribbons=False):
    # Code assumes coefficients will be named T_minus_N and T_N for pre and post-treatment estimates, respectively, and that -1 is the omitted estimate
    # If ribbons=True, a continuous ribbon is drawn, if False, point-wise error bars are drawn. Future versions of the function will draw ribbons using sup-t confidence bands and errorbars for pointwise confidence intervals

    # Extract coefficients, create confidence intervals, recover event times
    coef = model.coef().reset_index()
    se = model.se().reset_index()
    estimates = pd.merge(coef, se, on="Coefficient")
    estimates["lower_bound"] = estimates["Estimate"] - estimates["Std. Error"] * 1.96
    estimates["upper_bound"] = estimates["Estimate"] + estimates["Std. Error"] * 1.96
    estimates["event_time"] = (
        estimates["Coefficient"]
        .str.replace("minus_", "-", regex=False)
        .str.replace("T_", "", regex=False)
        ).astype(int)

    # Insert -1 row
    row = pd.DataFrame([{
        "Coefficient": "T_minus_1",
        "Estimate": 0.0,
        "Std. Error": 0.0,
        "lower_bound": 0.0,
        "upper_bound": 0.0,
        "event_time": -1
    }])
    estimates = pd.concat([estimates, row], ignore_index=True)
    estimates = estimates.sort_values(by="event_time")    

    event_study_plot, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=estimates, 
        x="event_time", 
        y="Estimate",  
        errorbar=None,
        ax=ax
    )
    sns.scatterplot(
        data=estimates, 
        x="event_time", 
        y="Estimate", 
        ax=ax
    )
    ax.axvline(
        x=-1, 
        linestyle="--", 
        color="gray", 
        linewidth=1,
        zorder=0
    )
    ax.axhline(
        y=0, 
        linestyle="--", 
        color="gray", 
        linewidth=1,
        zorder=0
    )

    if ribbons:
        ax.fill_between(
            x=estimates["event_time"],
            y1=estimates["lower_bound"],
            y2=estimates["upper_bound"],
            color="gray",
            alpha=0.2,
            zorder=0
        )
    else:
        ax.errorbar(
            x=estimates["event_time"],
            y=estimates["Estimate"],
            yerr=1.96 * estimates["Std. Error"],
            fmt="none",
            ecolor="black",
            capsize=3,
            alpha=0.5,
            zorder=0
        )

    ax.set_title("Event Study Estimates")
    ax.set_xlabel("Event Time")
    ax.set_ylabel("Coefficient")

    return event_study_plot, ax
