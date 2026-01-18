#!/usr/bin/env python3

# read 3 column data.csv:
# - name
# - size
# - speed
#
# make a log-log plot
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

data = pd.read_csv(
    "data.csv",
    names=["name", "size", "speed", "color", "lookup"],
    dtype={"name": str, "size": float, "speed": float, "color": str, "lookup": int},
    header=0,
)
# multiple speed by 1000
data["speed"] /= 1000

plt.figure(figsize=(7, 4))
for i, row in data.iterrows():
    plt.plot(
        row["size"],
        row["speed"],
        marker="o" if row["lookup"] == 0 else "*",
        ms=12 if row["lookup"] == 0 else 8,
        linestyle="None",
        # label=row["name"],
        color=row["color"],
        # base=2,
    )
formatter = plt.ScalarFormatter()
plt.gca().yaxis.set_major_formatter(formatter)

# set y-ax labels
# plt.yticks([0.5, 1, 2, 4, 8])
plt.ylim(0, 10)

plt.xlim(left=0)
plt.axvline(x=2.0, color="red", linestyle="-", linewidth=1)

plt.xlabel("Size (bits/bp)")
plt.ylabel("Speed (M reads/s)")
plt.title("FM-index exact read mapping size and count() throughput")
plt.grid(True, which="both", ls="--")
for i, row in data.iterrows():
    plt.text(row["size"] + 0.2, row["speed"] / 1.05, row["name"])

# Manually add legend handle for the star markers
plt.plot(
    [], [], marker="*", ms=8, linestyle="None", label="prefix lookup", color="black"
)
plt.legend(loc="upper left")


plt.savefig("comparison.png", bbox_inches="tight", dpi=300)
# plt.show()
