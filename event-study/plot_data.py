import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

panel = pd.read_csv("output/panel_data_simultaneous.csv")

# Plot group means for simultaneous treatment
# Note: sns can compute the group averages automatically,
# but it can't overlay dots unless the means are pre-computed
group_means = (
    panel
    .groupby(by=['time', 'ever_treated'])['y']
    .mean()
    .reset_index()
)

trend_plot, ax = plt.subplots(figsize=(8, 5))

plot_args = dict(
    x="time", 
    y="y", 
    hue="ever_treated", 
    palette=["silver", "seagreen"],
    ax=ax
)

sns.lineplot(data=panel, **plot_args)
sns.scatterplot(data=group_means, legend=False, **plot_args)

ax.set_title("Average Outcome Over Time by Treatment Group")
ax.set_xlabel("Time")
ax.set_ylabel("Mean of y")
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles, 
    labels=["Control", "Treatment"], 
    title="Treatment Status"
)

trend_plot.tight_layout()
plt.show()
trend_plot.savefig("output/treatment_trends_simultaneous.png", dpi=300)

# Plot group means for staggered treatment
# Skip the dots

panel = pd.read_csv("output/panel_data_staggered.csv")

panel["treatment_time_cat"] = panel["treatment_time"].astype("category")

trend_plot, ax = plt.subplots(figsize=(8, 5))

sns.lineplot(
    data=panel, 
    x="time", 
    y="y", 
    hue="treatment_time_cat", 
    ax=ax
)

ax.set_title("Average Outcome Over Time by Treatment Group")
ax.set_xlabel("Time")
ax.set_ylabel("Mean of y")

labs = (
    panel["treatment_time"]
    .copy()
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
    .astype(int)
    .astype(str)
)
labs = ["Never Treated" if x == "99999" else x for x in labs]

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles, 
    labels=labs, 
    title="Treatment Cohort"
)

trend_plot.tight_layout()
plt.show()
trend_plot.savefig("output/treatment_trends_staggered.png", dpi=300)

# Example using plotly (no CIs, unless computed by hand)
# trend_plotly = px.line(group_means, x='time', y='y', color='ever_treated',
#               title="Outcome Over Time by Treatment Group")
# trend_plotly.show()

