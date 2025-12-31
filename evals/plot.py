#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# df = pd.read_csv("results.csv")
# df = pd.read_csv("results-16.csv")
df = pd.read_csv("results-quad-6.csv")
sigma = df.sigma.unique()[0]
bits_per_symbol = 2 if sigma == 4 else 1
n = df.n.unique()[0]
size_mb = n * bits_per_symbol / 8 / (1000**2)  # in MB
pnglabel = f"{n//10**9}G" if n >= 10**9 else f"{n//10**6}M"

# if sigma == 4:
#     df.bits = df.bits / 2

# correct for quad->binary conversion size
df["extra_bits"] = df.bits - 1
df["overhead"] = 100 * df.extra_bits


def get_source(name):
    if "FullDouble" in name:
        return "double"
    if "TriBlock" in name:
        return "tri"
    if "Transposed" in name:
        return "fulltrans"
    if "Spider" in name:
        return "spider"
    if "Ranker" in name:
        return "quadrank"
    if "RS" in name:
        return "qwt"
    if "Rank9" in name:
        return "rank9"
    if "RankSmall" in name:
        return "ranksmall"
    if "TextWithRankSupport" in name:
        return "genedex"
    assert False


df["source"] = df["ranker"].apply(get_source)


cols = {
    "latency_1": ["Latency, 1 thread", "blue", 1],
    "loop_1": ["Loop, 1 thread", "green", 1],
    "stream_1": ["Stream, 1 thread", "red", 1],
    "latency_6": ["Latency, 6 threads", "blue", 6],
    "loop_6": ["Loop, 6 threads", "green", 6],
    "stream_6": ["Stream, 6 threads", "red", 6],
    "latency_12": ["Latency, 12 threads", "blue", 12],
    "loop_12": ["Loop, 12 threads", "green", 12],
    "stream_12": ["Stream, 12 threads", "red", 12],
}
rankers = {
    "quadrank": ["o", "#fcc007"],  # blog yellow
    "tri": ["o", "blue"],
    "fulltrans": ["o", "green"],
    "double": ["o", "black"],
    "qwt": ["s", "red"],
    "rank9": ["^", "green"],
    "ranksmall": ["x", "blue"],
    "genedex": ["+", "black"],
    "spider": ["*", "purple"],
}

ns = df.n.unique()
ns = [max(ns)]

# df = df[df.bits <= 1.35]

fig, ax = plt.subplots(3 * len(ns), 3, sharey=True, figsize=(15, 10 * len(ns)))
fig.tight_layout(pad=3.0)
for rr, n in enumerate(ns):
    # Make a 2D space-time scatter plot
    # Figure with two subfigures side-by-side
    df_n = df[df.n == n]
    for c, mode in enumerate(["latency", "loop", "stream"]):
        for r2, threads in enumerate([1, 6, 12]):
            r = rr * 2 + r2
            for index, row in df_n.iterrows():
                # print("row", row)
                marker, color = rankers[row["source"]]

                col = f"{mode}_{threads}"
                label, _color, t = cols[col]
                ax[r][c].scatter(
                    row["overhead"],
                    row[col],
                    color=color,
                    s=40,
                    marker=marker,
                    label=row["source"],
                )
            ax[r][c].set_axisbelow(True)
            suffix = "b" if sigma == 2 else "bp"
            if n >= 10**9:
                ax[r][c].set_title(f"n={n//10**9}G{suffix}, {size_mb/1000}GiB, {label}")
            else:
                ax[r][c].set_title(f"n={n//10**6}M{suffix}, {size_mb}MB, {label}")
            ax[r][c].set_xlabel("Overhead [%]")
            ax[r][c].set_ylabel("ns/query")
            ax[r][c].set_yscale("log", base=2)
            ax[r][c].set_xscale("log", base=2)
            xticks = [200, 100, 50, 25, 12.5, 6.25, 3.125, 1.5625]
            if sigma == 4:
                xticks = [100, 50, 25, 12.5, 6.25]
            ax[r][c].set_xticks(xticks)
            ax[r][c].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3g}"))
            ax[r][c].tick_params(axis="y", which="major", labelleft=True)
            ax[r][c].grid(True, which="major", ls="-", lw=0.3, axis="y")
            ax[r][c].get_yaxis().set_major_formatter(plt.ScalarFormatter())
            # ax[r][c].get_xaxis().set_major_formatter(plt.ScalarFormatter())
            if r == 0 and c == 0:
                handles, labels = ax[r][c].get_legend_handles_labels()
                unique = [
                    (h, l)
                    for i, (h, l) in enumerate(zip(handles, labels))
                    if l not in labels[:i]
                ]
                ax[r][c].legend(*zip(*unique))
            # draw hline
            if c == 0:
                lim = 80 / threads
            else:
                lim = 7.5 if r == 0 else 2.5
            ax[r][c].axhline(
                y=lim,
                color="red",
                linestyle="--",
                linewidth=1,
            )

fig.savefig(f"plot-{sigma}-{pnglabel}.png", bbox_inches="tight", dpi=300)
