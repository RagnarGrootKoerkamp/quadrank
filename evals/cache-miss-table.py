#!/usr/bin/env python3

import pandas as pd
import plot_rank

df = pd.read_csv("cache-misses-laptop.csv")
df["name"] = df.apply(
    lambda row: plot_rank.get_shortname(row["ranker"], row["sigma"])[1], axis=1
)
df["overhead"] = df.rel_size.apply(lambda x: (x - 1) * 100)
print(df)

print(
    df.pivot_table(
        index=["sigma", "name"],
        columns="count4",
        values=["overhead", "cache_misses_0"],
        sort=False,
    ),
)
