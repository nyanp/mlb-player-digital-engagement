# make features in parallel
import json
import multiprocessing
import os
import shutil
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset_helper import load_subdata
from src.feature import Context, get_features, get_fingerprint, normalize_feature_name, get_feature_path
from src.store import Store
from src.parser import make_df_base_from_train_engagement

_n_process_total = None
_feature_name = None
_data_dir = None
_base_dir = None
_feature_dir = None
_tmp_dir = None
_lag_requirements = None
_use_updated = True


def _base_df(data_dir: str, updated: bool):
    return make_df_base_from_train_engagement(load_subdata(data_dir, 'nextDayPlayerEngagement', updated))


def _make_feature(data_dir: str, output_dir: str, feature_name: str, index: int, in_total: int,
                  lag_requirements: int, use_updated: bool):
    from logging import getLogger
    import logging
    logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.DEBUG)
    logger = getLogger(__name__)
    fh = logging.FileHandler(f'log_{index}.log', 'w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.debug("start processing")

    # ピークをずらす
    time.sleep(index * 2)

    def _dec(v):
        try:
            if v is None:
                return v
            if type(v) in (np.float32, np.float64):
                return float(v)
            if type(v) in (np.int32, np.int64):
                return int(v)
            return v
        except Exception:
            return v

    try:

        base_df = _base_df(data_dir, use_updated)
        store = Store.train(data_dir, use_updated=use_updated)

        func = get_features()[feature_name]

        total_rows = len(base_df)
        stride = total_rows // in_total
        start = index * stride

        if index == in_total - 1:
            base_df = base_df.iloc[start:]
        else:
            base_df = base_df.iloc[start:start + stride]

        logger.debug(f"start feature generation, total {len(base_df)} rows")

        features = []
        for i, row in tqdm(base_df.iterrows()):
            date = row["dailyDataDate"]
            player = store.players[row["playerId"]].slice_until(date)

            ctx = Context(store, player, player.team, date,
                          store.daily_stats.slice_until(date), lag_requirements,
                          fallback_to_none=False)

            result = func(ctx)
            result = {k: _dec(v) for k, v in result.items()}
            features.append(result)

            if i % 10000 == 0:
                logger.debug(f"{i} / {len(base_df)}")

        logger.debug(f'finish creating features {index}')

        with open(os.path.join(output_dir, f'{feature_name}_{index}.json'), "w") as f:
            json.dump(features, f)
    except:
        import traceback
        logger.error('ERROR')
        logger.error(traceback.format_exc())


def _process(index: int):
    _make_feature(_data_dir, _tmp_dir, _feature_name, index, _n_process_total, _lag_requirements, _use_updated)


def _merge(tmp_dir: str, feature_name: str, n_process: int) -> pd.DataFrame:
    features = []

    for i in range(n_process):
        with open(os.path.join(tmp_dir, f'{feature_name}_{i}.json'), "r") as f:
            features.extend(json.load(f))

    return pd.DataFrame(features)


def make_feture_parallel(data_dir: str,
                         feature_name: str,
                         n_process: int = 32,
                         feature_store: str = '../features',
                         lag_requirements: int = 45,
                         use_updated: bool = True):
    global _n_process_total, _feature_name, _data_dir, _feature_dir, _tmp_dir, _lag_requirements, _use_updated
    fingerprint = get_fingerprint(_base_df(data_dir, use_updated))
    feature_name = normalize_feature_name(feature_name)

    _n_process_total = n_process
    _feature_name = feature_name
    _data_dir = data_dir
    _feature_dir = os.path.join(feature_store, fingerprint)
    _tmp_dir = os.path.join(_feature_dir, f'{feature_name}_tmp')
    _lag_requirements = lag_requirements
    _use_updated = use_updated

    os.makedirs(_tmp_dir, exist_ok=True)

    path = get_feature_path(feature_store, fingerprint, feature_name, lag_requirements)

    if os.path.exists(path):
        print(f'{path} already exists. skipped')
        return

    with multiprocessing.Pool(n_process) as p:
        p.map(_process, range(n_process))

    merged = _merge(_tmp_dir, feature_name, n_process)

    merged.to_feather(path)

    shutil.rmtree(_tmp_dir, ignore_errors=True)


def merge_feture_parallel(data_dir: str,
                          feature_name: str,
                          n_process: int = 32,
                          feature_store: str = '../features',
                          lag_requirements: int = 45,
                          use_updated: bool = True):
    fingerprint = get_fingerprint(_base_df(data_dir, use_updated))
    feature_name = normalize_feature_name(feature_name)
    _feature_dir = os.path.join(feature_store, fingerprint)
    _tmp_dir = os.path.join(_feature_dir, f'{feature_name}_tmp')
    merged = _merge(_tmp_dir, feature_name, n_process)
    path = get_feature_path(feature_store, fingerprint, feature_name, lag_requirements)
    merged.to_feather(path)
    shutil.rmtree(_tmp_dir, ignore_errors=True)
