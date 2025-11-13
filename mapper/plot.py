#!/usr/bin/env python3

# read 3 column data.csv:
# - name
# - size
# - speed
#
# make a log-log plot
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(
    "data.csv",
    names=["name", "size", "speed", "color"],
    dtype={"name": str, "size": float, "speed": float, "color": str},
    header=0,
)
print(data)
plt.figure(figsize=(7, 4))
for k, group in data.groupby("color"):
    plt.semilogy(
        group["size"],
        group["speed"],
        marker="o",
        ms=12,
        linestyle="None",
        label=k,
        color=k,
        base=2,
    )
plt.xlabel("Size (bits/bp)")
plt.ylabel("Speed (k reads/s)")
plt.title("FM-index exact read mapping size and throughput")
plt.grid(True, which="both", ls="--")
for i, row in data.iterrows():
    plt.text(row["size"] + 0.2, row["speed"] / 1.05, row["name"])
plt.savefig("comparison.png", bbox_inches="tight", dpi=300)
# plt.show()
