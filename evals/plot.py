#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from math import log


# Return (crate, name, symbol, oder)
def get_shortname(name):
    if sigma == 4:
        if "RSQ" in name:
            if "512" in name:
                return ("qwt", "qwt::RSQ512", "*", 0)
            return ("qwt", "qwt::RSQ256", "x", 2)
        if "FlatText" in name:
            if "512" in name:
                return ("genedex", "genedex::Flat512", "o", 7)
            if "64" in name:
                return ("genedex", "genedex::Flat64", "s", 8)
        if "Condensed" in name:
            if "512" in name:
                return ("genedex", "genedex::Condensed512", "*", 1)
            return ("genedex", "genedex::Condensed256", "v", 5)
        if "Ranker" in name:
            if "QuadBlock64" in name:
                return ("quadrank", "quadrank::QuadBlock64", "o", 6)
            if "QuadBlock24_8" in name:
                return ("quadrank", "quadrank::QuadBlock24_8", "d", 4)
            if "QuadBlock16" in name:
                return ("quadrank", "quadrank::QuadBlock16", "x", 3)
    if sigma == 2:
        if name == "RSNarrow":
            return ("qwt", "qwt::RSNarrow", "v", 14)
        if name == "RSWide":
            return ("qwt", "qwt::RSWide", "*", 6)
        if "Condensed" in name:
            if "512" in name:
                return ("genedex", "genedex::Condensed512", "x", 9)
            if "64" in name:
                return ("genedex", "genedex::Condensed64", "s", 16)
        if name == "RankSelect101111":
            return ("bitm", "bitm::RankSelect101111", "*", 5)
        if name == "Rank9":
            return ("rank9", "sux::Rank9", "v", 13)
        if "RankSmall" in name:
            if "2, 9" in name:
                return ("sux", "sux::RankSmall0", "v", 12)
            if "1, 9" in name:
                return ("sux", "sux::RankSmall1", "d", 10)
            if "1, 10" in name:
                return ("sux", "sux::RankSmall2", "x", 8)
            if "1, 11" in name:
                return ("sux", "sux::RankSmall3", "*", 3)
            if "3, 13" in name:
                return ("sux", "sux::RankSmall4", "+", 1)
        if "Ranker" in name:
            if "64x2" in name:
                return ("quadrank", "birank::BinaryBlock64x2", "o", 15)
            if "32x2" in name:
                return ("quadrank", "birank::BinaryBlock32x2", "d", 11)
            if "16x2" in name:
                return ("quadrank", "birank::BinaryBlock16x2", "x", 7)
            if "16>" in name:
                return ("quadrank", "birank::BinaryBlock16", "*", 2)
            if "Spider" in name:
                return ("spider", "birank::Spider", "*", 4)
    assert False


# TODO: use * symbol

blogyellow = "#fcc007"
styles = {
    "binary": ["x", "black"],
    "quadrank": ["o", blogyellow],
    "tri": ["o", "blue"],
    "fulltrans": ["o", "green"],
    "double": ["o", "pink"],
    "qwt": ["s", "red"],
    "rank9": ["^", "green"],
    "sux": ["x", "blue"],
    "genedex": ["+", "black"],
    "spider": ["*", "purple"],
    "bitm": ["v", "teal"],
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
    if x >= 2**30:
        return f"{x//2**30} Gi"
    if x >= 2**20:
        return f"{x//2**20} Mi"
    if x >= 2**10:
        return f"{x//2**10} Ki"
    return str(x)


def plot(ax, data, r, c, rows, mode, threads, plotn=False):
    col = f"{mode}_{threads}"
    plotlabel = {"latency": "Latency", "loop": "Loop", "stream": "Loop + prefetch"}[
        mode
    ] + f", {threads} thread{'s' if threads > 1 else ''}"

    maxn = data.n.max()

    groups = data.groupby(["ranker", "count4"], sort=False)

    for (ranker, count4), group in groups:
        (crate, shortname, marker, _order) = get_shortname(ranker)
        color = styles[crate][1]
        label = shortname

        if marker is None:
            continue

        # For scaling-by-n plots we only plot the count1 data.
        if plotn and count4 == 1:
            continue

        s = 40
        if sigma == 4 and not plotn and count4 == 0:
            label = None
            s = 10

        if color == blogyellow:
            s *= 1.5

        xvar = "size" if plotn else "overhead"

        if plotn:
            overhead = group[group.n == maxn]["overhead"].values[0]

            ls = "-"
            alpha = 1.0
            lw = 1.0
            ms = 3.5
            if color == blogyellow:
                ms *= 1.5**0.5

            if sigma == 2:
                if overhead < 5:
                    ls = "-"
                elif overhead < 15:
                    ls = "--"
                    alpha = 0.8
                else:
                    ls = ":"
                    alpha = 0.6
            else:
                if overhead < 10:
                    ls = "-"
                elif overhead < 25:
                    ls = "--"
                    alpha = 1.0
                elif overhead < 75:
                    ls = "-."
                    alpha = 1.0
                else:
                    ls = ":"
                    alpha = 0.8

            ax.plot(
                group[xvar],
                group[col],
                color=color,
                ms=ms,
                marker=marker,
                label=label,
                lw=lw,
                ls=ls,
                alpha=alpha,
            )
        else:
            ax.scatter(
                group[xvar],
                group[col],
                color=color,
                s=s,
                marker=marker,
                label=label,
            )

        if mode == "stream":
            col2 = f"loop_{threads}"
            if plotn:
                # ax.plot(
                #     group[xvar],
                #     group[col2],
                #     color=color,
                #     ms=s**0.5,
                #     marker=marker,
                #     # label="",
                #     alpha=0.1,
                #     lw=1,
                #     ls="-",
                # )
                pass
            else:
                ax.scatter(
                    group[xvar],
                    group[col2],
                    color=color,
                    s=s,
                    marker=marker,
                    # label="",
                    alpha=0.1,
                )
    ax.set_axisbelow(True)
    ax.set_title(plotlabel)
    if c == 0:
        ax.set_ylabel("ns/query")
    if r == rows - 1:
        if plotn:
            ax.set_xlabel("Size")
        else:
            ax.set_xlabel("Overhead [%]")
    ax.set_yscale("log", base=2)
    ax.set_xscale("log", base=2)
    if plotn:
        # xticks = [50, 25, 12.5, 6.25, 3.125, 1.5625]
        # ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3g}"))
        xticks = [2**i for i in [15, 20, 25, 30]]
        ax.set_xticks(xticks)
        labels = ["32 KiB", "1 MiB", "32 MiB", "1 GiB"]
        ax.set_xticklabels(labels)
        minor_xticks = [2**i for i in range(13, 33)]
        ax.set_xticks(minor_xticks, minor=True)
        # hide minor labels
        ax.tick_params(axis="x", which="minor", labelbottom=False)
        ax.grid(True, which="major", ls="-", lw=0.2, axis="x")
        pass
    else:
        xticks = [50, 25, 12.5, 6.25, 3.125, 1.5625]
        if sigma == 4:
            xticks = [200, 100, 50, 25, 12.5, 6.25]
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3g}"))
    ax.tick_params(axis="y", which="major", labelleft=True)
    ax.grid(True, which="major", ls="-", lw=0.3, axis="y")
    # Make sure all printed y-ax labels are integers.
    # TODO: format integers as normal (128) and decimals up to two periods (.2)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3g}"))
    if r == 0 and c == 0 and rows > 1:

        if sigma == 4:
            # Legend handle for small dot
            handles = [
                plt.Line2D(
                    [],
                    [],
                    color="black",
                    marker="o",
                    linestyle="",
                    markersize=40**0.5,
                    label="rank_all",
                ),
                plt.Line2D(
                    [],
                    [],
                    color="black",
                    marker="o",
                    linestyle="",
                    markersize=10**0.5,
                    label="rank_one",
                ),
            ]
            if plotn:
                # only rank_one
                handles = [
                    plt.Line2D(
                        [],
                        [],
                        color="black",
                        marker="o",
                        linestyle="",
                        markersize=40**0.5,
                        label="rank_one",
                    ),
                ]
            extra_legend = ax.legend(
                handles=handles,
                loc="lower left" if plotn else "lower right",
            )
            ax.legend()
            ax.add_artist(extra_legend)
        else:
            ax.legend(ncols=2, fontsize=8)

    if server:
        return
    # cache sizes and throughputs are tuned for laptop only.

    # draw hline for latency/throughput limit
    if mode == "latency":
        lim = 80 / threads
        ls = ":"
        lw = 1
    else:
        lim = 7.5 if r == 0 else 2.5
        ls = "--" if r == 0 else "-"
        lw = 0.5 if r == 0 else 0.5

    ax.axhline(
        y=lim,
        color="blue",
        linestyle=ls,
        linewidth=lw,
    )

    if plotn:
        lastrow = r == rows - 1
        # Cache sizes
        sizes = [("L1", 2**15), ("L2", 2**18), ("L3", 12 * 2**20)]
        y = 0.35 if sigma == 2 else 0.6
        for label, size in sizes:
            ax.axvline(
                x=size,
                color=(0, 0, 0, 0.7),
                linestyle="-",
                linewidth=0.5,
            )

            if lastrow:
                ax.text(
                    size / 1.2,
                    y,
                    label,
                    # rotation=90,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    color=(0, 0, 0, 0.7),
                )
        if lastrow:
            ax.text(
                size * 1.2,
                y,
                "RAM",
                # rotation=90,
                verticalalignment="bottom",
                horizontalalignment="left",
                color=(0, 0, 0, 0.7),
            )


def plot_small(plotn=False):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.33))
    fig.tight_layout(pad=3.0)

    if plotn:
        sub_df = df
    else:
        n = 2**20 // bits_per_symbol
        size_bytes = n * bits_per_symbol // 8
        sub_df = df[df.n == n]

    # Set title
    fig.suptitle(
        f"σ={sigma}"
        + ("" if plotn else f", n={format(n)}{unit}, size={format(size_bytes)}B"),
        fontsize=15,
        y=1.01,
    )

    plot(ax, sub_df, 0, 0, 1, "loop", 1, plotn=plotn)

    infix = "n" if plotn else "st"
    srv = "server" if server else "laptop"
    fig.savefig(
        f"plots/plot-{srv}-{infix}-{sigma}-small.png", bbox_inches="tight", dpi=300
    )


def plot_grid(plotn=False):
    # Make a 2D space-time scatter plot
    # Figure with two subfigures side-by-side

    n = sorted(df.n.unique())[-1]

    if plotn:
        sub_df = df
    else:
        size_bytes = n * bits_per_symbol // 8
        sub_df = df[df.n == n]

    fig, ax = plt.subplots(
        len(threads), 3, sharey="row", figsize=(15, 3.33 * len(threads))
    )
    fig.tight_layout(pad=1.5)

    # Set title
    if plotn:
        fig.suptitle(f"Inverse query throughput for σ={sigma}", fontsize=20, y=1.03)
    else:
        fig.suptitle(
            f"Space-time trade-off for σ={sigma}, n={format(n)}{unit}, size={format(size_bytes)}B",
            fontsize=20,
            y=1.03,
        )

    for c, mode in enumerate(["latency", "loop", "stream"]):
        for r, t in enumerate(threads):
            plot(ax[r][c], sub_df, r, c, len(threads), mode, t, plotn=plotn)

    if plotn:
        sizelabel = ""
    else:
        sizelabel = "-large"
    infix = "n" if plotn else "st"
    srv = "server" if server else "laptop"
    fig.savefig(
        f"plots/plot-{srv}-{infix}-{sigma}{sizelabel}.png", bbox_inches="tight", dpi=300
    )


# df_all = pd.read_csv(f"scaling-n.csv")
df_all = pd.read_csv(f"server.csv")

server = "latency_96" in df_all.columns
if server:
    threads = [1, 48, 96, 192]
else:
    threads = [1, 6, 12]
for sigma in [2, 4]:
    df = df_all[df_all.sigma == sigma].copy()

    assert df.sigma.unique() == [sigma]
    bits_per_symbol = 2 if sigma == 4 else 1
    unit = "b" if sigma == 2 else "bp"

    df["size"] = df["n"] * bits_per_symbol // 8
    df["overhead"] = 100 * (df.rel_size - 1)
    df["order"] = df["ranker"].apply(lambda name: get_shortname(name)[3])

    # sort by order
    df = df.sort_values(by=["order", "n"])

    df = df[df["size"] >= 2**13]

    plot_small()
    plot_grid()
    plot_grid(plotn=True)
