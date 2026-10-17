"""Loading the modelling dataset and the study's splits."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import PROCESSED, SEED
from .features import FEATURES

DATASET = PROCESSED / "dataset_v1.parquet"


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATASET)


def main_and_extra(df: pd.DataFrame):
    """Main experiments: rows whose league at t is one of the 14 fully covered leagues.
    Extra analysis: test-period rows from the 17 new leagues and from OTHER / UNKNOWN clubs.
    Category lists are kept identical in both frames so encodings line up."""
    core = df["league_group"] == "core"
    main = df[core].reset_index(drop=True)
    extra = df[~core & (df["split"] == "test")].reset_index(drop=True)
    return main, extra


def core_plus_unknown(df: pd.DataFrame) -> pd.DataFrame:
    """Variant training population: core-league rows plus rows whose club at t is unknown."""
    return df[df["league_group"].isin(["core", "UNKNOWN"])].reset_index(drop=True)


def random_split(df: pd.DataFrame, test_size=0.2, seed=SEED) -> pd.Series:
    """Ordinary random 80/20 row split (ignores time) - used ONLY to show how it over-states accuracy."""
    rng = np.random.default_rng(seed)
    return pd.Series(np.where(rng.random(len(df)) < test_size, "test", "train"), index=df.index)


def player_group_split(df: pd.DataFrame, test_size=0.2, seed=SEED) -> pd.Series:
    """Random 80/20 split by PLAYER with sklearn's GroupShuffleSplit (groups = player_id):
    all rows of a player land on one side. Still ignores time."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, test_idx = next(gss.split(df, groups=df["player_id"]))
    out = pd.Series("train", index=df.index)
    out.iloc[test_idx] = "test"
    return out


def xy(df: pd.DataFrame):
    """Features, and the training target = log change over 12 months (y - log value at t).
    Predicting the change (then adding back log_value) lets tree models handle any value level."""
    return df[FEATURES], (df["y"] - df["log_value"]).astype("float64")
