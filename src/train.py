import copy
import os
import shutil
import traceback
from typing import List, Dict, Union, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import mean_absolute_error

from src.config import Config
from src.feature import make_feature, get_column_list
from src.features.f900_second_order import get_second_order_features
from src.store import Store
from src.util import save_directory_as_kaggle_dataset, plot_importance, TimeSeriesSplit


def save_file(path, filename=None):
    filename = filename or os.path.basename(path)
    shutil.copy(
        path,
        os.path.join(wandb.run.dir, filename),
    )
    wandb.save(os.path.join(wandb.run.dir, filename))


TARGET_COLS = ['target1', 'target2', 'target3', 'target4']
CAT_COLS = [
    'player_statusCode',
    'player_transactions_typeCode',
    'primaryPositionCode',
    'birthCountry',
    'team_game_gameType'
]


def _mask_new_user(store: Store, df_train: pd.DataFrame):
    player_df = store.players.player_df
    players_for_tests = set(player_df.index[player_df.playerForTestSetAndFuturePreds == True].values)
    mask = df_train['playerId'].isin(players_for_tests).values
    return mask


def get_mask_by_season_df(season_df: pd.DataFrame, df_train: pd.DataFrame):
    # parse season information
    season_df['seasonStartDate'] = pd.to_datetime(season_df['seasonStartDate'])
    season_df['seasonEndDate'] = pd.to_datetime(season_df['seasonEndDate'])
    ts = df_train['dailyDataDate']

    mask = None
    for i, season in season_df.iterrows():
        if i == 0:
            mask = (ts >= season.seasonStartDate) & (ts <= season.seasonEndDate)
        else:
            mask = mask | ((ts >= season.seasonStartDate) & (ts <= season.seasonEndDate))
    return mask.values


def to_float32(X: pd.DataFrame):
    X_tr = X.copy()
    for c in X_tr.columns:
        X_tr[c] = X_tr[c].astype(np.float32)

    return X_tr


def make_cv_splits(gap_days: int, train_days: int, lag_req: int, validation_start_times: List = None,
                   use_updated: bool = True):
    if validation_start_times is None:
        if use_updated:
            validation_start_times = ['2020-08-01', '2021-05-20', '2021-06-20']
        else:
            validation_start_times = ['2019-08-01', '2020-08-01', '2021-04-01']

    def _train_end(validation_start_time):
        return np.datetime64(validation_start_time) - np.timedelta64(gap_days, 'D')

    def _valid_range(start_time):
        # use fix time-period for validation
        return start_time, np.datetime64(start_time) + np.timedelta64(30, 'D')
        #return start_time, np.datetime64(start_time) + np.timedelta64(lag_req + 2, 'D'),

    def _train_start(validation_start_time):
        return _train_end(validation_start_time) - np.timedelta64(train_days, 'D')

    def _split(validation_start_time):
        return (_train_start(validation_start_time), _train_end(validation_start_time)), _valid_range(
            validation_start_time)

    return [_split(t) for t in validation_start_times]


def lgb_cv(params: Dict,
           cv: TimeSeriesSplit,
           df_train: pd.DataFrame,
           store: Store,
           X: pd.DataFrame,
           Ys: pd.DataFrame,
           tgt: str,
           config: Config,
           sample_weight_func=None):
    importances = []
    maes = []
    iterations = []
    oof_prediction = np.zeros(len(X))
    valid_indices = []

    for cv_idx, (train_idx, valid_idx) in enumerate(cv.split(df_train)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = Ys[tgt].iloc[train_idx], Ys[tgt].iloc[valid_idx]

        print(valid_idx)
        print(f"fold {cv_idx} train: {X_tr.shape}, valid: {X_va.shape}")

        if not config.train_new_users_only:
            # train_new_users_only=Falseの場合でも、validation dataはnew usersで行う。
            df_train_va = df_train.iloc[valid_idx]
            mask = _mask_new_user(store, df_train_va)
            X_va = X_va[mask]
            y_va = y_va[mask]
            masked_valid_idx = valid_idx[mask]
        else:
            masked_valid_idx = valid_idx

        model = lgb.LGBMRegressor(**params)

        if sample_weight_func is not None:
            sample_weight = sample_weight_func(df_train.iloc[train_idx])
        elif config.weight_new_users:
            df_train_tr = df_train.iloc[train_idx]
            mask = _mask_new_user(store, df_train_tr)
            sample_weight = (mask + 1) / 2.0
        else:
            sample_weight = None

        cat_cols = [c for c in CAT_COLS if c in X_tr.columns]

        X_tr = to_float32(X_tr)
        X_va = to_float32(X_va)

        print(f"cv {cv_idx} cols: {X_tr.columns}")

        model.fit(X_tr, y_tr,
                  sample_weight=sample_weight,
                  categorical_feature=cat_cols,
                  eval_set=[(X_va, y_va)],
                  early_stopping_rounds=100,
                  verbose=200)

        predicted = model.predict(X_va)
        oof_prediction[masked_valid_idx] = predicted
        valid_indices.append(masked_valid_idx)

        df = pd.DataFrame()
        df['feature'] = list(X_tr.columns)
        df['importance'] = model.booster_.feature_importance(importance_type='gain')

        importances.append(df)
        mae = mean_absolute_error(y_va, predicted)
        maes.append(mae)

        iterations.append(model.best_iteration_)

    return importances, maes, iterations, oof_prediction, np.concatenate(valid_indices)


def train_single_lag(params: Union[Dict, List[Dict]],
                     df_train: pd.DataFrame,
                     store: Store,
                     features: List[str],
                     lag_req: int,
                     train_config: Config,
                     season_df: pd.DataFrame = None,
                     run=None,
                     log_config: bool = False,
                     sample_weight_func=None,
                     extra_df: pd.DataFrame = None,
                     output_dir: str = 'artifacts',
                     num_seeds: int = 1,
                     drop_features: Optional[List[str]] = None):
    print(f"lag: {lag_req}")

    if run is None:
        run = wandb.init(project='mlb', entity='nyanp')

    config = wandb.config

    df_train = df_train.copy()

    if train_config.features_per_target is not None:
        features_all = list(features)
        for feats in train_config.features_per_target.values():
            for f in feats:
                if f not in features_all:
                    features_all.append(f)
    else:
        features_all = features
    X_orig = make_feature(df_train, store, with_target=True, feature_list=features_all, lag_requirements=lag_req,
                          second_order_feature=train_config.second_order_features)

    if extra_df is not None:
        assert train_config.extra_df_on

    if not isinstance(params, list):
        params = [params] * 4

    if train_config.train_new_users_only:
        mask = _mask_new_user(store, df_train)
        df_train = df_train[mask].reset_index(drop=True)
        X_orig = X_orig[mask].reset_index(drop=True)
        if extra_df is not None:
            extra_df = extra_df[mask].reset_index(drop=True)
        print(f'[train_new_users_only]training data reduced to {len(X_orig)}')

    if train_config.on_season_only:
        if season_df is not None:
            mask = get_mask_by_season_df(season_df, df_train)
        else:
            mask = df_train['dailyDataDate'].dt.month.isin([3, 4, 5, 6, 7, 8, 9]).values
        df_train = df_train[mask].reset_index(drop=True)
        X_orig = X_orig[mask].reset_index(drop=True)
        if extra_df is not None:
            extra_df = extra_df[mask].reset_index(drop=True)
        print(f'[on_season_only]training data reduced to {len(X_orig)}')

    Ys = X_orig[TARGET_COLS]
    X_orig.drop(TARGET_COLS, axis=1, inplace=True)

    if train_config.select_features:
        raise NotImplementedError()

    # 4-weeks
    print(f'lag requirements: {config.lag_requirements}')
    print(f'gap days: {config.gap_days}')
    print(f'shape: {X_orig.shape}')

    splits = make_cv_splits(train_config.gap_days, train_config.train_days, lag_req, use_updated=store.use_updated)
    split_for_train = ('2015-10-01', '2021-10-01')

    if log_config:
        config.features = list(X_orig.columns)
        config.n_features = len(X_orig.columns)
        config.update({f'cv_period{i}': splits[i] for i in range(len(splits))})

    all_metrics = np.zeros((len(splits), len(TARGET_COLS)))

    for tgt_index, tgt in enumerate(TARGET_COLS):
        cv = TimeSeriesSplit('dailyDataDate', splits)

        current_feature_set = list(features)
        if train_config.features_per_target is not None and train_config.features_per_target.get(tgt):
            for f in train_config.features_per_target[tgt]:
                assert f not in current_feature_set, f"extra feature {f} already exists"
                current_feature_set.append(f)

        if extra_df is not None and tgt in train_config.extra_df_on:
            assert len(X_orig) == len(extra_df)
            X_tgt = pd.concat([X_orig, extra_df], axis=1).copy()
            extra_cols = list(extra_df.columns)
        else:
            X_tgt = X_orig.copy()
            extra_cols = []

        current_columns = get_column_list(current_feature_set) + extra_cols

        if train_config.second_order_features:
            current_columns += get_second_order_features()

        if drop_features is not None:
            if isinstance(drop_features, dict):
                drop_features_tgt = drop_features.get(tgt, [])
            else:
                drop_features_tgt = drop_features
            current_columns = [c for c in current_columns if c not in drop_features_tgt]

        importances, maes, iterations, oof, valid_indices = lgb_cv(params[tgt_index], cv,
                                                                   df_train, store,
                                                                   X_tgt[current_columns],
                                                                   Ys, tgt,
                                                                   train_config,
                                                                   sample_weight_func)
        np.save(os.path.join(output_dir, f"{tgt}_lag{lag_req}_oof.npy"), oof)
        np.save(os.path.join(output_dir, f"{tgt}_lag{lag_req}_indices.npy"), valid_indices)

        for cv_idx in range(len(splits)):
            wandb.run.summary[f"{tgt}_cv{cv_idx}_lag{lag_req}"] = all_metrics[cv_idx, tgt_index] = maes[cv_idx]

        iteration_for_test = int(1.1 * np.mean(iterations))
        wandb.run.summary[f"{tgt}_lag{lag_req}"] = np.mean(maes)
        wandb.run.summary[f"{tgt}_lag{lag_req}_iterations"] = iteration_for_test

        if train_config.train_full:
            for seed_round in range(num_seeds):
                params_train = copy.deepcopy(params[tgt_index])
                params_train['n_estimators'] = iteration_for_test

                if seed_round > 0:
                    params_train['random_state'] = 2021 + seed_round

                clf = lgb.LGBMRegressor(**params_train)

                start = pd.to_datetime(split_for_train[0])
                end = pd.to_datetime(split_for_train[1])
                mask = (df_train['dailyDataDate'] >= start) & (df_train['dailyDataDate'] < end)
                clf.fit(to_float32(X_tgt[mask][current_columns]), Ys[tgt][mask],
                        categorical_feature=[c for c in CAT_COLS if c in current_columns])

                if num_seeds == 1:
                    model_path = os.path.join(output_dir, f'model_{tgt}_lag{lag_req}.bin')
                else:
                    model_path = os.path.join(output_dir, f'model_{tgt}_lag{lag_req}_seed{seed_round}.bin')
                clf.booster_.save_model(model_path)
                save_file(model_path)

        try:
            plot_importance(pd.concat(importances), f"importance_{tgt}_lag{lag_req}.png")
            save_file(f"importance_{tgt}_lag{lag_req}.png")
        except Exception:
            print(f"save artifacts failed.")
            print(traceback.format_exc())

    for i in range(len(splits)):
        run.summary[f'cv{i}_lag{lag_req}'] = np.mean(all_metrics, axis=1)[i]

    run.summary[f'lag{lag_req}'] = np.mean(all_metrics)


def train(params,
          df_train_orig: pd.DataFrame,
          store: Store,
          train_config: Config,
          run_name=None,
          season_df=None,
          sample_weight_func=None,
          extra_df: pd.DataFrame = None,
          output_dir: str = 'artifacts',
          num_seeds: int = 1,
          upload_dir: Optional[str] = None):
    os.makedirs(output_dir, exist_ok=True)
    upload_dir = upload_dir or output_dir
    if train_config.upload:
        os.makedirs(upload_dir, exist_ok=True)

    try:
        lag_requirements = train_config.lags or [45]
        run = wandb.init(project='mlb', entity='nyanp')
        config = wandb.config

        if run_name:
            run.name = run_name

        if not isinstance(params, list):
            params = [params] * 4

        config.params1 = params[0]
        config.params2 = params[1]
        config.params3 = params[2]
        config.params4 = params[3]
        config.feature_set = list(train_config.features)
        config.train_full = train_config.train_full
        config.train_new_users_only = train_config.train_new_users_only
        config.lag_requirements = lag_requirements
        config.gap_days = train_config.gap_days
        config.select_features = train_config.select_features
        config.on_season_only = train_config.on_season_only
        config.train_days = train_config.train_days
        config.trim_by_season_df = season_df is not None
        config.sample_weight_func = sample_weight_func.__name__ if sample_weight_func is not None else None
        config.features_per_lag = train_config.features_per_lag
        config.features_per_target = train_config.features_per_target
        config.weight_new_users = train_config.weight_new_users
        config.num_seeds = num_seeds
        config.use_updated = store.use_updated
        config.drop_features = train_config.drop_features
        config.second_order_features = train_config.second_order_features

        if extra_df is not None:
            config.extra_cols = list(extra_df.columns)
            config.extra_df_on = train_config.extra_df_on

        for lag_index, lag_req in enumerate(lag_requirements):
            feature_set = list(train_config.features)
            if train_config.features_per_lag is not None:
                feature_set += train_config.features_per_lag.get(lag_req, [])
            train_single_lag(params, df_train_orig, store, feature_set, lag_req, train_config,
                             season_df=season_df, run=run,
                             log_config=lag_index == 0,
                             sample_weight_func=sample_weight_func,
                             extra_df=extra_df,
                             output_dir=output_dir,
                             num_seeds=num_seeds,
                             drop_features=config.drop_features)
        wandb.join()

        if train_config.upload:
            # copy wandb's artifacts to output directory
            shutil.copy(os.path.join(wandb.run.dir, 'config.yaml'), output_dir)
            shutil.copy(os.path.join(wandb.run.dir, 'wandb-summary.json'), output_dir)
            shutil.copy(os.path.join(wandb.run.dir, 'wandb-metadata.json'), output_dir)
            # uplooad
            save_directory_as_kaggle_dataset(upload_dir, "mlb dataset", "mlb-dataset")
    except Exception:
        print(traceback.format_exc())
        raise
