import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import dir_data, dir_plots
from itertools import product

# Load data
data_path = dir_data / "20251128_Elicitrevised.csv"
save_path = dir_plots / "fig2_eeg_gait_heatmap_pub.png"
df = pd.read_csv(data_path, sep=";")
print(f"Loaded {len(df)} studies from {data_path.name}\n")

# Clean column names 
df.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)

# Explode multi-value columns
rows = []
for _, row in df.iterrows():
    eeg_types = [e.strip() for e in str(row.get("Type_of_EEG_electrodes", "")).split(";") if e.strip()]
    gait_systems = [g.strip() for g in str(row.get("Gait_measurement_system", "")).split(";") if g.strip()]
    for eeg, gait in product(eeg_types, gait_systems):
        rows.append({"Type_of_EEG_electrodes": eeg, "Gait_measurement_system": gait})

df_expanded = pd.DataFrame(rows)

# Pivot for heatmap
heat_data = df_expanded.groupby(
    ["Type_of_EEG_electrodes", "Gait_measurement_system"]
).size().unstack(fill_value=0)

# Plot heatmap
plt.figure(figsize=(16, 9)) 
sns.heatmap(
    heat_data,
    annot=True,
    fmt="d",
    cmap="OrRd",         
    linewidths=1.0,
    linecolor="gray",
    cbar=False,
    annot_kws={"size": 26, "weight": "bold"} 
)

# Titles and labels
plt.title(
    "EEG Electrode Types vs Gait Measurement Systems",
    fontsize=26,
    weight='bold',
    pad=20
)
plt.ylabel("Type of EEG Electrodes", fontsize=20, weight='bold')
plt.xlabel("Gait Measurement System", fontsize=20, weight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(save_path, dpi=600, bbox_inches="tight")
plt.show()
print(f"High-resolution heatmap saved to: {save_path}")

# Descriptive statistics
print("Number of studies per EEG electrode type:")
print(df_expanded["Type_of_EEG_electrodes"].value_counts(), "\n")

print("Number of studies per gait measurement system:")
print(df_expanded["Gait_measurement_system"].value_counts(), "\n")

print("EEG electrode type vs gait measurement system (cross-tabulation):")
print(heat_data, "\n")

print("Percentage distribution (row-wise):")
heat_percent = heat_data.div(heat_data.sum(axis=1), axis=0) * 100
print(heat_percent.round(1))

# Descriptive statistics with percentages 
print("Number of studies per EEG electrode type:")
counts_eeg = df_expanded["Type_of_EEG_electrodes"].value_counts()
percent_eeg = df_expanded["Type_of_EEG_electrodes"].value_counts(normalize=True) * 100
df_eeg_stats = pd.DataFrame({"Count": counts_eeg, "Percentage (%)": percent_eeg.round(1)})
print(df_eeg_stats, "\n")

print("Number of studies per gait measurement system:")
counts_gait = df_expanded["Gait_measurement_system"].value_counts()
percent_gait = df_expanded["Gait_measurement_system"].value_counts(normalize=True) * 100
df_gait_stats = pd.DataFrame({"Count": counts_gait, "Percentage (%)": percent_gait.round(1)})
print(df_gait_stats, "\n")

print("EEG electrode type vs gait measurement system (cross-tabulation with percentages):")
cross_tab = df_expanded.groupby(["Type_of_EEG_electrodes", "Gait_measurement_system"]).size().unstack(fill_value=0)
cross_tab_percent = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
cross_tab_percent = cross_tab_percent.round(1)
cross_tab_combined = cross_tab.astype(str) + " (" + cross_tab_percent.astype(str) + "%)"
print(cross_tab_combined)

