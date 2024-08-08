from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import json
import os
from pandas import Index
import re
from scipy.stats import linregress
import sys
from scipy.stats import linregress
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import numpy.core.multiarray
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from skimage.filters import threshold_li, threshold_otsu
from tqdm import tqdm
from typing_extensions import Literal

# DATA = ROOT / "traffic_results"
DATA = ROOT / "cc_results/traffic_results_subset"


def get_score_info(df: DataFrame, selected: Series) -> DataFrame:
    """
    `df` will look something like:

                         feat  score
                 latitude_NAN  485.0
                longitude_NAN  130.0
             vehicle_year_NAN  139.0
                  reg_lat_NAN  446.0
                 reg_long_NAN  119.0
                          ...    ...
           vehicle_type_other  425.0
    vehicle_type_recreational   75.0
         vehicle_type_trailer   68.0
           vehicle_type_truck  624.0
             vehicle_type_nan    0.0

    [144 rows x 2 columns]

    We want to identify if the sex or race features (if present) have high
    importances.
    """
    # get max and median importances for feature groups of interest
    feature_clusters = {
        "locs": df["feat"][
            df["feat"].str.match(".*lat.*|.*long.*|.*outstate.*|.*reg_km.*")
        ].values.tolist(),
        "vhcls": df["feat"][df["feat"].str.match(".*vehicle.*")].values.tolist(),
        "race": df["feat"][df["feat"].str.match(".*race.*")].values.tolist(),
        "sex": df["feat"][df["feat"].str.match(".*sex.*")].values.tolist(),
        "time": df["feat"][
            df["feat"].str.match(".*hour.*|.*year.*|.*month.*|.*week.*")
        ].values.tolist(),
        "patrol": df["feat"][df["feat"].str.match(".*entity.*")].values.tolist(),
        "agency": df["feat"][df["feat"].str.match(".*agency.*")].values.tolist(),
    }
    infos = []
    for cluster, features in feature_clusters.items():
        dff = df[df["feat"].isin(features)].sort_values(by="score", ascending=False)
        if len(dff) == 0:
            fname = ""
            fmax = np.nan
            fmed = np.nan
        else:
            fname = dff["feat"].iloc[0]
            fmax = dff["score"].iloc[0]
            fmed = dff["score"].median()

        infos.append(
            DataFrame(
                {
                    cluster: fname,
                    f"{cluster}_max": fmax,
                    f"{cluster}_med": fmed,
                },
                index=[0],
            )
        )
    info = pd.concat(infos, axis=1)
    return info


def read_report_tables(file: Path) -> tuple[Series, DataFrame]:
    report = file.read_text()
    selected = None
    lines = report.split("\n")
    for line in lines:
        if line.startswith("["):
            selected = Series(data=eval(line))
    if selected is None:
        raise ValueError(f"Could not parse selected features in {file}")

    table = None
    start = 0
    stop = None
    for i, line in enumerate(lines):
        if line.startswith("| "):
            start = i + 2
            i = start
            # for wrapper reports, don't read until end
            while i < len(lines):
                line = lines[i]
                if not line.startswith("|"):
                    stop = i
                    break
                i += 1
            break

    table = lines[start:stop]
    if table is None or len(table) < 1:
        raise ValueError(f"Could not parse feature scores table in {file}")
    feats, scores = [], []
    try:
        for line in table:
            feat, score = line[1:-1].split("|")
            feat = str(feat.strip())
            score = float(score.strip())
            feats.append(feat)
            scores.append(score)
    except ValueError as e:
        tab = "\n".join(table[:10] + table[-10:])
        print(f"Got error: {e} for table:\n{tab}")

    df = DataFrame(data={"feat": feats, "score": scores})
    df["feat"] = df["feat"].astype(str)
    return selected, df


def load_embed_metrics(path: Path, model: Literal["linear", "lgbm"]) -> DataFrame:
    select_dir = path.parents[1] / "selection"
    embed_dir = select_dir / "embed"
    file = embed_dir / f"{model}_embedded_selection_report.md"
    selected, df = read_report_tables(file)
    df = get_score_info(df, selected)
    df["selection"] = f"embed_{model}"
    return df


def load_wrap_metrics(path: Path, model: Literal["linear", "lgbm"]) -> DataFrame:
    select_dir = path.parents[1] / "selection"
    wrap_dir = select_dir / "wrapper"
    file = wrap_dir / "wrapper_selection_report.md"
    selected, df = read_report_tables(file)
    df = get_score_info(df, selected)
    df["selection"] = "wrap"
    return df


def load_assoc_metrics(path: Path) -> DataFrame:
    file = path.parents[1] / "selection/filter/association_selection_report.md"
    selected, df = read_report_tables(file)
    df = get_score_info(df, selected)
    df["selection"] = "assoc"
    return df


def load_pred_metrics(path: Path) -> DataFrame:
    file = path.parents[1] / "selection/filter/prediction_selection_report.md"
    selected, df = read_report_tables(file)
    df = get_score_info(df, selected)
    df["selection"] = "pred"
    return df


def info_from_path(path: Path) -> DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.drop(columns=["trainset", "5-fold"])
    df = df.pivot_table(
        index="metric",
        columns=["model", "selection", "embed_selector"],
        values="holdout",
    ).T.reset_index()
    df.rename_axis(None, axis="columns", inplace=True)

    df_lgbm = load_embed_metrics(path, "lgbm")
    # df_linear = load_embed_metrics(path, "linear")
    # df_wrap = load_wrap_metrics(path, model="linear")
    df_assoc = load_assoc_metrics(path)
    # df_pred = load_pred_metrics(path)
    df_feats = pd.concat([df_lgbm, df_assoc], axis=0)
    # df_feats = df_lgbm

    target_info = path.parents[3].stem
    target, feats = target_info.split("__")

    df["feats"] = feats
    df["target"] = target
    df["info"] = target_info
    df["path"] = path
    df = pd.merge(df, df_feats, how="left", on="selection")
    return df


def load_summary_df() -> DataFrame:
    paths = sorted(DATA.rglob("final_performances.csv"))
    dfs = []
    dfs = Parallel(n_jobs=-1)(
        delayed(info_from_path)(path)
        for path in tqdm(paths, desc="Loading info from disk...")
    )

    df = pd.concat(dfs, axis=0, ignore_index=True)
    target = df["target"].copy()
    df.drop(columns="target", inplace=True)
    df.insert(1, "target", target)
    # df = df[~df.model.isin(["dummy"])].copy()
    print(df.to_markdown(tablefmt="simple", index=False))
    # df.index = Index(name="rid", data=df["info"])
    index_cols = ["info", "model", "selection"]
    df.index = pd.MultiIndex.from_frame(df[index_cols], names=index_cols)

    df.drop(columns=["path", "info"], inplace=True)
    return df


def nan_slope(x: Series, y: Series) -> float:
    if len(x.unique()) == 1 or len(y.unique()) == 1:
        return np.nan
    return linregress(x, y)[0]


def mean_diffs(x: Series, y: Series) -> float:
    idx_present = x == 1
    return y[idx_present].mean() - y[~idx_present].mean()


def print_bias_corrs(df: DataFrame, method: Literal["spearman", "pearson"]) -> None:
    bias_cols = ["incl_sex", "incl_race"]
    all_metrics = ["acc", "auroc", "f1", "npv", "ppv", "sens", "spec"]
    metrics = ["acc", "auroc", "f1"]
    print(
        f"{method.capitalize()} correlations between performance metrics and inclusion of sensitive features."
    )
    df = df.copy().drop(columns="bal-acc", errors="ignore")
    corrs = df.groupby("target").corr(method=method, numeric_only=True)[bias_cols]

    slopes = []
    diffs = []
    targs = []

    for targ, dfs in df.groupby("target"):
        targ_slopes = []
        targ_diffs = []
        for indicator in bias_cols:
            ms = cast(
                DataFrame,
                (
                    dfs[metrics]  # type: ignore
                    .apply(lambda s: nan_slope(dfs[indicator], s))
                    .to_frame()
                ),
            )
            ms.columns = Index(name="bias", data=[indicator])
            targ_slopes.append(ms)

            ds = cast(
                DataFrame,
                (
                    dfs[metrics]  # type: ignore
                    .apply(lambda s: mean_diffs(dfs[indicator], s))
                    .to_frame()
                ),
            )
            ds.columns = Index(name="bias", data=[indicator])
            targ_diffs.append(ds)

        # add targets to indexes
        targ_slopes = pd.concat(targ_slopes, axis=1)
        targ_slopes = pd.concat({targ: targ_slopes}, names=["target"])
        targ_diffs = pd.concat(targ_diffs, axis=1)
        targ_diffs = pd.concat({targ: targ_diffs}, names=["target"])

        slopes.append(targ_slopes)
        diffs.append(targ_diffs)
        targs.append(targ)
    slopes = pd.concat(slopes, axis=0)
    diffs = pd.concat(diffs, axis=0)
    # of course slopes and diffs are identical in this formulation since the
    # indicators are binary...

    targets = df["target"].unique().tolist()
    # metrics = ["acc", "auroc", "f1", "npv", "ppv", "sens", "spec"]
    sel = pd.MultiIndex.from_product([targets, metrics])
    # corrs = (
    #     df.corr(method=method, numeric_only=True)
    #     .loc[bias_cols]
    #     .T.round(3)
    #     .drop(index=bias_cols)
    # ).round(3)
    # covs = (
    #     dfo.cov(numeric_only=True)[["incl_sex", "incl_race"]]
    #     .iloc[2:]
    #     .map(lambda x: f"{x:1.2e}")
    # )

    info = pd.concat(
        [corrs, diffs, slopes], axis=1, keys=["corr", "𝜇_diff", "slope"]
    ).loc[sel]
    print(info.round(5))

    # print(corrs.to_markdown(tablefmt="simple"))
    # print(covs.to_markdown(tablefmt="simple", floatfmt="1.2e"))


if __name__ == "__main__":
    pd.options.display.max_colwidth = 20
    df = load_summary_df()
    # df = df[df["target"] == "outcome"]
    df.insert(4, "incl_sex", df["feats"].isin(["race+sex", "norace"]).astype(int))
    df.insert(5, "incl_race", df["feats"].isin(["race+sex", "nosex"]).astype(int))

    print(df)
    dummy = df[df["model"] == "dummy"]
    dummy.index = dummy.index.droplevel("model")
    lgbm = df[df["model"] == "lgbm"]
    lgbm.index = lgbm.index.droplevel("model")

    ix_start = df.columns.tolist().index("acc")
    ix_stop = df.columns.tolist().index("spec") + 1
    metrics = df.columns.tolist()[ix_start:ix_stop]
    if not (lgbm[metrics] > dummy[metrics]).all().all():
        raise ValueError("Must filter on models worse than dummy.")

    # dff = df[df["selection"] == "embed_lgbm"].drop(columns=["path", "info"])
    dff = df.drop(columns=["path", "info"], errors="ignore")
    # df_sel = dff.reset_index(drop=True).drop(columns=["model", "embed_selector"]).T.copy()
    # df_sel.columns = df_sel.loc["feats"]
    # df_sel = df_sel.drop(index="feats")

    dfo = df.iloc[:, :15].reset_index(drop=True)
    print("\nAll results")
    print("=" * 81)
    print(
        dfo.sort_values(by=["target", "acc"], ascending=False)
        .round(3)
        .to_markdown(tablefmt="simple", index=False)
    )

    df_best = (
        dfo[dfo["model"] != "dummy"]
        .groupby(["target", "model"])[dfo.columns]
        .apply(lambda grp: grp.nlargest(4, "acc"), include_groups=True)
        .reset_index(drop=True)
        .drop(columns=["embed_selector", "feats", "bal-acc"])
        .round(3)
    )
    print(
        "\nBest 4 results (i.e. best selection methods) for each combination of model and target"
    )
    print("=" * 81)
    print(df_best)

    df_dummy = (
        dfo[dfo["model"] == "dummy"]
        .groupby(["target"])[dfo.columns]
        .apply(lambda grp: grp.nlargest(1, "acc"), include_groups=True)
        .reset_index(drop=True)
        .drop(columns=["embed_selector", "feats", "bal-acc"])
        .round(3)
    )
    print("\nDummy performances:")
    print("=" * 81)
    print(df_dummy)

    bias_cols = ["bias", "incl_sex", "incl_race"]

    print("\nEffect of Sensitive Features on Performance: All Results")
    print("=" * 81)
    print_bias_corrs(dfo[dfo["model"] != "dummy"], "spearman")
    # print_bias_corrs(dfo, "pearson")

    for N in [4, 5]:
        # see comment in https://stackoverflow.com/a/78582800 for the dumb
        # [dfo.columns] subsetting below. Only way to include groups and not
        # get the warning...
        df_best = (
            dfo[dfo["model"] != "dummy"]
            .groupby(["target", "model"])[dfo.columns]
            .apply(lambda grp: grp.nlargest(N, "acc"), include_groups=True)
            .reset_index(drop=True)
        )
        print(f"\nEffect of Sensitive Features on Performance: Best {N} Runs")
        print("=" * 81)
        print_bias_corrs(df_best, "spearman")
        # print_bias_corrs(dfo, "pearson")

    # dfo = dfo[dfo["selection"].isin(["none"])]
    # print("\nCorrelations when using no feature selection")
    # print("=" * 81)
    # print_bias_corrs(dfo, "spearman")
