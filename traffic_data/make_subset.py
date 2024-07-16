from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
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
import pandas as pd
import random
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal
from math import ceil

from sklearn.model_selection import StratifiedShuffleSplit


DATA = ROOT / "traffic_data"
BASE = DATA / "traffic_data_processed.parquet"
OUT = DATA / "traffic_data_processed_subset.parquet"

TARGETS = ["outcome", "violation_type", "search_conducted", "chrg_title_mtch"]
# remove outcome, since it dupes violation_type
STRATIFY = ["violation_type", "search_conducted", "chrg_title_mtch"]
SEED = 69

"""
>>> df.outcome.isna().mean()
np.float64(0.38944331144877964)

I.e. the "outcome" variable is about 40% NaNs, and so a reduction to 60% of the
dataset size does seem to complete on Niagara. To be safe, we reduce to a third.
"""

if __name__ == "__main__":
    random.seed(SEED)
    df = pd.read_parquet(BASE)
    n_third = ceil(len(df) / 3)
    ss = StratifiedShuffleSplit(n_splits=1, train_size=n_third, random_state=SEED)
    ix = next(ss.split(X=df, y=df[STRATIFY]))[0]
    dfs = df.iloc[ix]
    df.to_parquet(OUT)
    print(dfs)
    print(f"Saved reduced subset to {OUT}")
