import os
from typing import Callable, List

import pandas as pd
from pandas.util import hash_pandas_object
from tqdm import tqdm

from src.features.f000_basic import *
from src.features.f100_lag import *
from src.features.f200_sns import *
from src.features.f300_daily_stats import *
from src.features.f400_player_agg import *
from src.features.f500_team_agg import *
from src.features.f600_datetime import *
from src.features.f700_events_agg import *
from src.features.f800_meta import *
from src.features.f900_second_order import *
from src.store import Store

MINIMUM_LAG = 45


def get_fingerprint(df):
    head = f"d{len(df)}"
    if len(df) == 2506176:
        head = "train"
    h = hash(frozenset(hash_pandas_object(df)))
    hash_value = f"{h:X}"[:6]
    return f"{head}_{hash_value}_{df.index[0] - 20000000}_{df.index[-1] - 20000000}"


def get_feature_path(base_dir: str, fingerprint:str, fname: str, lag_requirements: int):
    stem = normalize_feature_name(fname)
    postfix = f'_{lag_requirements}' if is_parameterized(fname) else ''
    return os.path.join(base_dir, fingerprint, f"{stem}{postfix}.f")


def select_features(df: pd.DataFrame, feature_list: List[str], with_target=True):
    if with_target and 'target' not in feature_list:
        feature_list = list(feature_list) + ['target']

    schema = get_feature_schema()
    columns = []
    for f in feature_list:
        for c in schema[f]:
            assert c in df.columns, f'feature {c}(from {f}) not found in the dataframe'
        columns += schema[f]

    return df[columns]


def load_feature(base_df: pd.DataFrame,
                 feature_list: List[str],
                 feature_store: str = '../features',
                 lag_requirement: int = 45):
    fingerprint = get_fingerprint(base_df)

    feature_list = [normalize_feature_name(k) for k in feature_list]

    dfs = []
    for f in feature_list:
        path = get_feature_path(feature_store, fingerprint, f, lag_requirement)
        dfs.append(pd.read_feather(path))

    return pd.concat(dfs, axis=1)


def make_feature(base_df: pd.DataFrame,
                 store: Store,
                 feature_list: List[str] = None,
                 feature_store: str = '../features',
                 load_from_store: bool = True,
                 save_to_store: bool = True,
                 debug=True,
                 with_target=False,
                 lag_requirements:int=45,
                 fallback_to_none:bool=True,
                 second_order_feature:bool=False):
    fingerprint = get_fingerprint(base_df)

    if feature_list:
        feature_list = {normalize_feature_name(k): get_feature(k) for k in feature_list}
    else:
        feature_list = get_features()

    if with_target and 'f999_target' not in feature_list:
        feature_list['f999_target'] = get_features()['f999_target']

    feature_paths = {
        fname: get_feature_path(feature_store, fingerprint, fname, lag_requirements) for fname in feature_list.keys()
    }

    if load_from_store:
        feature_list_to_calc = {
            k: v for k, v in feature_list.items() if not os.path.exists(feature_paths[k])
        }
        feature_list_from_cache = {
            k: v for k, v in feature_list.items() if k not in feature_list_to_calc
        }
    else:
        feature_list_to_calc = feature_list
        feature_list_from_cache = {}

    if not with_target:
        if "f999_target" in feature_list_to_calc:
            del feature_list_to_calc["f999_target"]
        if "f999_target" in feature_list_from_cache:
            del feature_list_from_cache["f999_target"]

    feature_to_cols = {}

    schema = get_feature_schema()

    print(f"calculate: {list(feature_list_to_calc.keys())}")
    print(f"from cache: {list(feature_list_from_cache.keys())}")

    dfs = []

    if feature_list_to_calc:
        features = []
        for i, row in tqdm(base_df.iterrows()):
            feature = {}
            date = row["dailyDataDate"]
            player = store.players[row["playerId"]].slice_until(date)

            ctx = Context(store, player, player.team, date,
                          store.daily_stats.slice_until(date),
                          lag_requirements, fallback_to_none=fallback_to_none)

            for fname, func in feature_list_to_calc.items():
                result = func(ctx)
                for k in result.keys():
                    if k in feature:
                        raise ValueError(f"feature name {k} is duplicated across features")
                if fname not in feature_to_cols:
                    feature_to_cols[fname] = list(result.keys())

                if debug:
                    schema_f = schema[fname]
                    # check column schema
                    for c in feature_to_cols[fname]:
                        assert c in result, f"column schema inconsistent in feature {fname}"
                        assert c in schema_f, f"column schema mismatch, expected: {schema_f}, actual: {c}"
                    for c in schema_f:
                        assert c in feature_to_cols[fname], f"column schema mismatch. {c} not found in generated feature"
                    for c in result.keys():
                        assert c in feature_to_cols[fname], f"column schema inconsistent in feature {fname}"
                feature.update(result)

            features.append(feature)
        features_df = pd.DataFrame(features)
        dfs.append(features_df)

        if save_to_store:
            os.makedirs(os.path.join(feature_store, fingerprint), exist_ok=True)
            for fname in feature_list_to_calc.keys():
                features_df[feature_to_cols[fname]].to_feather(feature_paths[fname])

    if feature_list_from_cache:
        dfs += [pd.read_feather(feature_paths[fname]) for fname in feature_list_from_cache.keys()]

    assert len(dfs)
    dst = pd.concat(dfs, axis=1)

    if second_order_feature:
        dst = f900_player_box_score_rank(dst, base_df)

    return dst


def generate_features(base_df: pd.DataFrame,
                      store: Store,
                      feature_list: List[str] = None,
                      feature_store: str = '../features',
                      lag_requirements: int = 45):
    feature_list = feature_list or get_feature_list()
    fingerprint = get_fingerprint(base_df)

    feature_paths = {
        fname: get_feature_path(feature_store, fingerprint, fname, lag_requirements) for fname in feature_list
    }

    for f in feature_list:
        if os.path.exists(feature_paths[f]):
            print(f'feature {f} already exists. skipped')
            continue

        try:
            make_feature(base_df, store, [f], feature_store)
        except:
            print(f'error in generating feature {f}')
            import traceback
            print(traceback.format_exc())
            pass
