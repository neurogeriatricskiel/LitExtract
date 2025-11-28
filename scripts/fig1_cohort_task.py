import pandas as pd
import matplotlib.pyplot as plt
from utils.config import dir_data, dir_plots

# Load data
data_path = dir_data / "20251128_Elicitrevised.csv"
save_path = dir_plots / "fig1_cohort_task.png"

df = pd.read_csv(data_path, sep=";")
df.columns = df.columns.str.strip().str.replace(r"[\s\-]+", "_", regex=True)
df = df.dropna(subset=["Cohort", "Gait_Task"])

# Cohort order
order = [
    "HA",
    "PwPD",
    "HA & PwPD",
    "Non-PD clinical cohorts",
    "HA & non-PD clinical cohorts"
]

# Pivot
pivot = (
    df.pivot_table(index="Cohort", columns="Gait_Task",
                   values="Citation", aggfunc="count", fill_value=0)
      .reindex(order)
)

# Flip for correct visual ordering
pivot_plot = pivot.iloc[::-1]

# Plot with publication-quality styling
fig, ax = plt.subplots(figsize=(12, 5), dpi=800)

pivot_plot.plot(
    kind="barh",
    stacked=True,
    colormap="tab20",
    edgecolor="none",
    ax=ax
)

ax.set_title("Cohort vs Gait Task Distribution", fontsize=20, weight="bold", pad=20)
ax.set_xlabel("Number of Studies", fontsize=18)
ax.set_ylabel("Cohort", fontsize=18)
ax.tick_params(axis="both", labelsize=18)

# Legend label cleanup
label_map = {
    "Overground walking": "Only Overground walking",
    "Treadmill walking": "Only Treadmill walking"
}
handles, labels = ax.get_legend_handles_labels()
labels = [label_map.get(l, l) for l in labels]

ax.legend(
    handles, labels,
    title="Gait Task",
    fontsize=14,
    title_fontsize=16,
    framealpha=0.8,
    loc="center right",
    bbox_to_anchor=(1.0, 0.55)
)

# Grid
ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)

plt.tight_layout()
plt.savefig(save_path, dpi=600, bbox_inches="tight")
plt.show()

print(f"Plot saved to: {save_path}\n")

# Descriptive statistics

print("Descriptive Statistics\n")

cohort_counts = df["Cohort"].value_counts(dropna=False)
print("Number of studies per cohort:")
print(cohort_counts, "\n")

task_counts = df["Gait_Task"].value_counts(dropna=False)
print("Number of studies per gait task:")
print(task_counts, "\n")

cohort_gait_table = pd.crosstab(df["Cohort"], df["Gait_Task"])
print("Cohort vs Gait Task (cross-tabulation):")
print(cohort_gait_table, "\n")

cohort_gait_percent = cohort_gait_table.div(cohort_gait_table.sum(axis=1), axis=0) * 100
print("Percentage distribution (row-wise):")
print(cohort_gait_percent.round(1))
