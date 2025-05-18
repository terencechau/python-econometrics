import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

panel = pd.read_csv("output/panel_data.csv")

# Plot group means for ever vs never treated
trend_plot, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=panel, x="time", y="y", hue="ever_treated", ax=ax)

ax.set_title("Average Outcome Over Time by Treatment Group")
ax.set_xlabel("Time")
ax.set_ylabel("Mean of y")
ax.legend(title="Group")

trend_plot.tight_layout()
trend_plot.show()
trend_plot.savefig("output/treatment_trends.png", dpi=300)

# Using plotly (no CIs, unless computed by hand)
group_means = (
    panel
    .groupby(by=['time', 'ever_treated'])['y']
    .mean()
    .reset_index()
)

trend_plotly = px.line(group_means, x='time', y='y', color='ever_treated',
              title="Outcome Over Time by Treatment Group")
trend_plotly.show()

