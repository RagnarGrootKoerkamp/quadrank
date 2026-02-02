#!/usr/bin/env python3
# Mostly copied from plot.py
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import colorsys


# Return (crate, name, symbol, oder)
def get_shortname(name):
    if "RSQ" in name:
        if "512" in name:
            return ("qwt", "QuadFm<qwt::RSQ512>", "*", 9)
        return ("qwt", "QuadFm<qwt::RSQ256>", "x", 10)
    if "genedex::" in name:
        if "FlatText" in name:
            if "512" in name:
                return ("genedex-fm", "Genedex<Flat512>", "o", 3)
            if "64" in name:
                return ("genedex-fm", "Genedex<Flat64>", "s", 4)
        if "Condensed" in name:
            if "512" in name:
                return ("genedex-fm", "Genedex<Condensed512>", "*", 1)
            return ("genedex-fm", "Genedex<Condensed256>", "v", 2)
        assert False

    if "FlatText" in name:
        if "512" in name:
            return ("genedex", "QuadFm<genedex::Flat512>", "o", 7)
        if "64" in name:
            return ("genedex", "QuadFm<genedex::Flat64>", "s", 8)
    if "Condensed" in name:
        if "512" in name:
            return ("genedex", "QuadFm<genedex::Condensed512>", "*", 5)
        return ("genedex", "QuadFm<genedex::Condensed256>", "v", 6)
    if "Ranker" in name:
        if "QuadBlock64" in name:
            return ("quadrank", "QuadFm<QuadRank64>", "o", 13)
        if "QuadBlock24_8" in name:
            return ("quadrank", "QuadFm<QuadRank24_8>", "d", 12)
        if "QuadBlock16" in name:
            return ("quadrank", "QuadFm<QuadRank16>", "x", 11)
    if name == "AWRY":
        return ("awry", "AWRY", "^", 10.5)
    assert False, f"unknown name {name}"


blogyellow = "#fcc007"
styles = {
    "quadrank": blogyellow,
    "qwt": "red",
    "genedex": "black",
    "genedex-fm": "blue",
    "awry": "purple",
}


def plot(ax, data, r, c, rows, mode, threads, plotn=False, small=False):
    subdata = data[(data["mode"] == mode) & (data["threads"] == threads)]
    plotlabel = {
        "Sequential": "Sequential",
        "Batch": "Batch",
        "Prefetch": "Batch + prefetch",
    }[mode] + f", {threads} thread{'s' if threads > 1 else ''}"

    groups = subdata.groupby("name", sort=False)

    for name, group in groups:
        (crate, shortname, marker, _order) = get_shortname(name)
        color = styles[crate]
        label = shortname

        if marker is None:
            continue

        s = 60

        ax.scatter(
            group["bits_per_bp"],
            # group["ns_per_read"],
            group["Mreads_per_sec"],
            color=color,
            s=s,
            marker=marker,
            label=label,
        )
    if mode == "Prefetch":
        subdata = data[(data["mode"] == "Batch") & (data["threads"] == threads)]
        groups = subdata.groupby("name", sort=False)
        for name, group in groups:
            (crate, shortname, marker, _order) = get_shortname(name)
            color = styles[crate]
            label = shortname

            if marker is None:
                continue

            s = 60

            ax.scatter(
                group["bits_per_bp"],
                # group["ns_per_read"],
                group["Mreads_per_sec"],
                color=color,
                s=s,
                marker=marker,
                alpha=0.1,
            )
    ax.set_axisbelow(True)
    ax.set_title(plotlabel)
    if c == 0:
        # ax.set_ylabel("ns/query")
        ax.set_ylabel("Mreads/sec")
    if r == rows - 1:
        ax.set_xlabel("Size (bits/bp)")
    ax.set_yscale("log", base=2)
    # xticks = [50, 25, 12.5, 6.25, 3.125, 1.5625]
    # ax.set_xticks(xticks)
    # ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3g}"))
    ax.tick_params(axis="y", which="major", labelleft=True)
    ax.grid(True, which="major", ls="-", lw=0.5, axis="y", color="black", alpha=0.5)

    # set minor y ticks in between powers of 2
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=2.0**0.5, subs=[0.5], numticks=10)
    )
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, which="minor", ls="--", lw=0.5, axis="y", color="black", alpha=0.3)
    # Make sure all printed y-ax labels are integers.
    # TODO: format integers as normal (128) and decimals up to two periods (.2)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2g}"))

    ax.set_xlim(xmin=0, xmax=7)
    ax.axvline(x=2.0, color="black", linestyle="--", linewidth=0.5)


def add_legend(axs, plotn, small=False):
    fig = axs[0][0].get_figure()
    handles, labels = axs[0][0].get_legend_handles_labels()
    rows = 4
    pos = (0.5, -0.19)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=(len(handles) + rows - 1) // rows,
        bbox_to_anchor=pos,
    )


def plot_grid(plotn=False):
    scale = 0.7
    fig, ax = plt.subplots(
        len(threads), 3, sharey="row", figsize=(15 * scale, 3.33 * len(threads) * scale)
    )
    fig.tight_layout(pad=1.5)

    # Set title
    fig.suptitle(
        "FM-index count throughput",
        fontsize=20,
        y=1.05,
    )

    for c, mode in enumerate(["Sequential", "Batch", "Prefetch"]):
        for r, t in enumerate(threads):
            plot(ax[r][c], df, r, c, len(threads), mode, t, plotn=plotn)
    add_legend(ax, plotn)

    fig.savefig(f"plots/plot-fm.png", bbox_inches="tight", dpi=300)


df = pd.read_csv(f"fm.csv")
input_len = 3117292070
df["bits_per_bp"] = df["bytes"] * 8 / input_len
df["order"] = df["name"].apply(lambda name: get_shortname(name)[3])
df["ns_per_read"] = 1e9 / df["reads_per_sec"]
df["Mreads_per_sec"] = df["reads_per_sec"] / 1e6
threads = [1, 6, 12]
df = df.sort_values(by=["order"])
plot_grid()
