import copy
import traceback
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, List

import numpy as np

from src.player import Player
from src.store import Store
from src.streamdf import TimeSeriesStream
from src.team import Team

_ALL_FEATURES = {}
_FEATURE_COLUMNS = {}
_LAG_PARAMETERIZED = {}
_ALL_FEATURE_NAMES = set()


@dataclass
class Context:
    store: Store
    player: Player
    team: Team
    daily_data_date: np.datetime64
    daily_stats: TimeSeriesStream
    lag_requirements: int
    current_feature_name: str = None
    fallback_to_none: bool = True


def feature(columns: List[str], lag_parameterized: bool = False):
    def _feature(func):
        _ALL_FEATURE_NAMES.add(func.__name__)

        _FEATURE_COLUMNS[func.__name__] = columns
        _LAG_PARAMETERIZED[func.__name__] = lag_parameterized

        prefix = _prefix(func.__name__)
        features_with_same_prefix = [f for f in _ALL_FEATURE_NAMES if _prefix(f) == prefix]
        assert len(features_with_same_prefix) == 1, f"feature prefix is duplicated! {features_with_same_prefix}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            ctx = args[0]
            assert isinstance(ctx, Context)
            assert len(args) == 1
            ctx.current_feature_name = func.__name__

            try:
                return func(ctx)
            except Exception:
                msg = f"WARNING: exception occured in feature {func.__name__}: {traceback.format_exc()}"
                warnings.warn(msg)
                print(msg)
                # guard
                if ctx.fallback_to_none:
                    return empty_feature(func.__name__)
                else:
                    raise

        _ALL_FEATURES[func.__name__] = wrapper

        return wrapper

    return _feature


def is_parameterized(fname: str) -> bool:
    return _LAG_PARAMETERIZED[normalize_feature_name(fname)]


def _prefix(feature_name: str) -> str:
    return feature_name.split('_')[0]


def get_features() -> Dict[str, Callable]:
    return _ALL_FEATURES


def get_feature_list() -> List[str]:
    return list(sorted(_ALL_FEATURES.keys()))


def get_feature_schema() -> Dict[str, List[str]]:
    return copy.deepcopy(_FEATURE_COLUMNS)


def get_column_list(features: List[str]) -> List[str]:
    cols = []
    for f in features:
        cols += _FEATURE_COLUMNS[normalize_feature_name(f)]

    assert len(cols) == len(list(set(cols))), "feature name duplicates!"
    return cols


def empty_feature(name: str) -> Dict:
    return {k: None for k in _FEATURE_COLUMNS[name]}


def get_feature(name: str) -> Callable:
    return _ALL_FEATURES[normalize_feature_name(name)]


def normalize_feature_name(name: str) -> str:
    if name in _ALL_FEATURES:
        return name

    for k in _ALL_FEATURES.keys():
        if _prefix(k) == name:
            return k

    raise ValueError(f"feature {name} not found")


@feature(['target1', 'target2', 'target3', 'target4'])
def f999_target(ctx: Context) -> Dict:
    return {
        'target1': ctx.player.engagement['target1'][-1],
        'target2': ctx.player.engagement['target2'][-1],
        'target3': ctx.player.engagement['target3'][-1],
        'target4': ctx.player.engagement['target4'][-1]
    }
