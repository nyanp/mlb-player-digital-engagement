import os
import pickle
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.nn import predict_nn, MLP, CNN, Union


def get_mix_ratio(lag: int, target: str):
    # gbdt, cnn, mlp
    if lag == 0:
        return [1, 0, 0]

    if target == 'target1':
        return [2, 1, 1]

    if target == 'target2':
        if lag < 14:
            return [4, 1, 1]
        else:
            return [2, 1, 1]

    if target == 'target3':
        return [1, 1, 1]

    if target == 'target4':
        return [8, 1, 1]

    return [1, 0, 0]


def ensemble(lag, pred_gbdt, pred_mlp, pred_cnn):
    if pred_mlp is None:
        return pred_gbdt  # no nn

    if pred_gbdt is None:
        return (pred_mlp + pred_cnn) / 2  # nn only

    # mixed
    mixed = np.zeros_like(pred_gbdt)

    for i in range(4):
        tgt = f"target{i + 1}"
        w = get_mix_ratio(lag, tgt)
        mixed[:, i] = (w[0] * pred_gbdt[:, i] + w[1] * pred_cnn[:, i] + w[2] * pred_mlp[:, i]) / (w[0] + w[1] + w[2])

    return mixed


class EnsembleModel:
    def __init__(self, models: List[lgb.Booster], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights

        features = list(self.models[0].feature_name())

        for m in self.models[1:]:
            assert features == list(m.feature_name())

    def predict(self, x):
        predicted = np.zeros((len(x), len(self.models)))

        for i, m in enumerate(self.models):
            w = self.weights[i] if self.weights is not None else 1
            predicted[:, i] = w * m.predict(x)

        ttl = np.sum(self.weights) if self.weights is not None else len(self.models)
        return np.sum(predicted, axis=1) / ttl

    def feature_name(self) -> List[str]:
        return self.models[0].feature_name()


class NNInferenceModel:
    def __init__(self, models: List[Union[MLP, CNN]], scaler: StandardScaler, features: List[str], device):
        self.models = models
        self.scaler = scaler
        self.features = features
        self.device = device

    def predict(self, pred_df: pd.DataFrame) -> np.ndarray:
        return predict_nn(pred_df[self.features], self.models, self.scaler, self.device)


class LagManager:
    def __init__(self, last_ts: np.datetime64, lag_requirements: List[int]):
        self.last_ts = last_ts
        self.lag_requirements = list(sorted(lag_requirements))
        print(f"Modelbank last_ts: {self.last_ts}, models: {self.lag_requirements}")

    def get_current_lag(self, date: np.datetime64, verbose=False) -> int:
        gap_days = (date - self.last_ts) / np.timedelta64(1, 'D')

        if verbose:
            print(f"date: {date}, gap_days: {gap_days}")

        for lag in self.lag_requirements:
            if gap_days <= lag + 1:
                if verbose:
                    print(f"use lag={lag},")
                return lag

        return self.lag_requirements[-1]


class ModelBank:
    def __init__(self,
                 model_path: str,
                 lag_requirements: List[int],
                 last_timestamp_of_training_data: np.datetime64,
                 nn_metadata: Optional[Dict],
                 num_seeds: int = 1):
        self.model_path = model_path
        self.nn_metadata = nn_metadata
        self.last_ts = last_timestamp_of_training_data

        self.lag2gbdts = {}  # type: Dict[int, List[lgb.Booster]]
        self.lag2cnn = {}  # type: Dict[int, List[CNN]]
        self.lag2mlp = {}  # type: Dict[int, List[MLP]]
        self.lag2scaler = {}  # type: Dict[int, StandardScaler]
        self.device = None
        self.gbdt_lags = LagManager(last_timestamp_of_training_data, lag_requirements)
        self.nn_lags = None  # type: Optional[LagManager]

        for lag in lag_requirements:
            self.lag2gbdts[lag] = []
            for i in range(4):
                if num_seeds == 1:
                    booster = lgb.Booster(
                        model_file=os.path.join(model_path, 'gbdt_new', f'model_target{i + 1}_lag{lag}.bin'))
                else:
                    booster = EnsembleModel(
                        [lgb.Booster(model_file=os.path.join(model_path, 'gbdt_new',
                                                             f'model_target{i + 1}_lag{lag}_seed{s}.bin')) for s in
                         range(num_seeds)] +
                        [lgb.Booster(model_file=os.path.join(model_path, 'gbdt_new2',
                                                             f'model_target{i + 1}_lag{lag}_seed{s}.bin')) for s in
                         range(num_seeds)]
                    )
                self.lag2gbdts[lag].append(booster)

        if nn_metadata is not None:
            print(f"Load NN")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            lag_requirements_nn = list(sorted([int(k) for k in nn_metadata['models'].keys()]))

            self.nn_lags = LagManager(last_timestamp_of_training_data, lag_requirements_nn)

            for k, meta in nn_metadata['models'].items():
                lag = int(k)
                self.lag2mlp[lag] = [
                    torch.load(os.path.join(model_path, 'nn_new', path), self.device) for path in meta['mlp_model_path']
                ]
                self.lag2cnn[lag] = [
                    torch.load(os.path.join(model_path, 'nn_new', path), self.device) for path in meta['cnn_model_path']
                ]
                with open(os.path.join(model_path, 'nn_new', meta['pkl_path'][0]), "rb") as f:
                    self.lag2scaler[lag] = pickle.load(f)
        else:
            print(f"Inference without NN")

    def get_gbdt_models(self, date: np.datetime64, verbose: bool = False) -> List[lgb.Booster]:
        """
        obtain best leak-free models
        :param date:
        """
        return self.lag2gbdts[self.gbdt_lags.get_current_lag(date, verbose)]

    def get_cnn_model(self, date: np.datetime64, verbose: bool = False) -> NNInferenceModel:
        """
        obtain best leak-free models
        :param date:
        """
        lag = self.nn_lags.get_current_lag(date, verbose)
        return NNInferenceModel(self.lag2cnn[lag],
                                self.lag2scaler[lag],
                                self.nn_metadata['models'][str(lag)]['columns'],
                                self.device)

    def get_mlp_model(self, date: np.datetime64, verbose: bool = False) -> NNInferenceModel:
        """
        obtain best leak-free models
        :param date:
        """
        lag = self.nn_lags.get_current_lag(date, verbose)
        return NNInferenceModel(self.lag2mlp[lag],
                                self.lag2scaler[lag],
                                self.nn_metadata['models'][str(lag)]['columns'],
                                self.device)

    def get_current_features(self, i: int, date: np.datetime64) -> List[str]:
        models = self.get_gbdt_models(date)
        features = models[i].feature_name()
        return features

    def get_current_nn_features(self, date: np.datetime64) -> List[str]:
        lag = self.nn_lags.get_current_lag(date)
        return self.nn_metadata['models'][str(lag)]['columns']
