#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# df = pd.read_csv("binary.csv")
df = pd.read_csv("quad.csv")
sigma = df.sigma.unique()[0]
bits_per_symbol = 2 if sigma == 4 else 1
n = df.n.unique()[0]
unit = "b" if sigma == 2 else "bp"

# if sigma == 4:
#     df.bits = df.bits / 2

# correct for quad->binary conversion size
df["extra_bits"] = df.bits - 1
df["overhead"] = 100 * df.extra_bits

has_count4 = "count4" in df.columns


def get_source(name):
    if "<Plain" in name:
        return "skip"
    if "FullDouble" in name:
        return "quadrank"
    if "TriBlock" in name:
        return "quadrank"
    if "Transposed" in name:
        return "quadrank"
    if "Spider" in name:
        return "spider"
    if "Ranker" in name:
        if "Block3" in name:
            return "skip"
        if "Block5" in name:
            return "skip"
        return "quadrank"
    if "RS" in name:
        return "qwt"
    if "Rank9Sel" in name:
        return "skip"
    if "Rank9<Vec" in name:
        return "skip"
    if "Rank9" in name:
        return "rank9"
    if "RankSmall" in name:
        return "ranksmall"
    # if "FlatTextWithRank" in name:
    #     return "skip"
    if "TextWithRankSupport" in name:
        return "genedex"
    if "Jacobson" in name:
        return "skip"
    if "RsDict" in name:
        return "skip"
    if "RankSelect10" in name:
        return "bitm"
    if "RsVec" in name:
        return "skip"
    if "RankSelect" in name:
        return "skip"
    if "RankSimple" in name:
        return "skip"
    return name


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
styles = {
    "quadrank": ["o", "#fcc007"],  # blog yellow
    "tri": ["o", "blue"],
    "fulltrans": ["o", "green"],
    "double": ["o", "pink"],
    "qwt": ["s", "red"],
    "rank9": ["^", "green"],
    "ranksmall": ["x", "blue"],
    "genedex": ["+", "black"],
    "spider": ["*", "purple"],
    "bitm": ["v", "brown"],
    "skip": [None, None],
    "other": ["*", "black"],
}


def get_style(name):
    if name in styles:
        return styles[name]
    # next colour in palette
    color = plt.get_cmap("tab10")(len(styles) % 10)
    styles[name] = [".", color]
    return styles[name]


def format(x):
    if x >= 10**9:
        return f"{x//10**9}G"
    if x >= 10**6:
        return f"{x//10**6}M"
    if x >= 10**3:
        return f"{x//10**3}K"
    return str(x)


def plot(ax, data, r, c, rows, mode, threads):
    col = f"{mode}_{threads}"
    label, _color, _t = cols[col]
    for _index, row in data.iterrows():
        # print("row", row)
        marker, color = get_style(row["source"])
        if marker is None:
            continue

        if not has_count4 or row["count4"] == 1:
            s = 40
        else:
            s = 10

        ax.scatter(
            row["overhead"],
            row[col],
            color=color,
            s=s,
            marker=marker,
            label=row["source"],
        )

        if mode == "stream":
            col2 = f"loop_{threads}"
            ax.scatter(
                row["overhead"],
                row[col2],
                color=color,
                s=s,
                marker=marker,
                # label="",
                alpha=0.1,
            )
    ax.set_axisbelow(True)
    ax.set_title(label)
    if c == 0:
        ax.set_ylabel("ns/query")
    if r == rows - 1:
        ax.set_xlabel("Overhead [%]")
    ax.set_yscale("log", base=2)
    ax.set_xscale("log", base=2)
    xticks = [50, 25, 12.5, 6.25, 3.125, 1.5625]
    if sigma == 4:
        xticks = [200, 100, 50, 25, 12.5, 6.25]
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3g}"))
    ax.tick_params(axis="y", which="major", labelleft=True)
    ax.grid(True, which="major", ls="-", lw=0.3, axis="y")
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    # ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if r == 0 and c == 0:
        handles, labels = ax.get_legend_handles_labels()
        unique = [
            (h, l)
            for i, (h, l) in enumerate(zip(handles, labels))
            if l not in labels[:i]
        ]
        ax.legend(*zip(*unique))
    # draw hline
    if mode == "latency":
        lim = 80 / threads
    else:
        lim = 7.5 if r == 0 else 2.5
    ax.axhline(
        y=lim,
        color="red",
        linestyle="--",
        linewidth=1,
    )


def plot_small():
    n = 1024000 // bits_per_symbol
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.33))
    fig.tight_layout(pad=3.0)

    size_bytes = n * bits_per_symbol // 8
    print(n, size_bytes)

    # Set title
    fig.suptitle(
        f"σ={sigma}, n={format(n)}{unit}, size={format(size_bytes)}B",
        fontsize=15,
        y=1.01,
    )

    df_n = df[df.n == n]
    plot(ax, df_n, 0, 0, 1, "loop", 1)

    fig.savefig(f"plot-{sigma}-small.png", bbox_inches="tight", dpi=300)


def plot_grid(n):
    # Make a 2D space-time scatter plot
    # Figure with two subfigures side-by-side
    df_n = df[df.n == n]

    fig, ax = plt.subplots(3, 3, sharey=True, figsize=(15, 10))
    fig.tight_layout(pad=3.0)

    size_bytes = n * bits_per_symbol // 8
    title = f"n={format(n)}{unit}, size={format(size_bytes)}B"

    # Set title
    fig.suptitle(f"Space-time trade-off for σ={sigma}, {title}", fontsize=20, y=1.01)

    for c, mode in enumerate(["latency", "loop", "stream"]):
        for r, threads in enumerate([1, 6, 12]):
            plot(ax[r][c], df_n, r, c, 3, mode, threads)

    pnglabel = f"{n//10**9}G" if n >= 10**9 else f"{n//10**6}M"
    fig.savefig(f"plot-{sigma}-{pnglabel}.png", bbox_inches="tight", dpi=300)


plot_small()
plot_grid(32_000_000_000 // bits_per_symbol)
# plot_grid(64_000_000_000)
