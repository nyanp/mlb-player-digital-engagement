from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class Config:
    lags: List[int]
    features: List[str]
    features_per_lag: Optional[Dict[int, List[str]]]=None
    features_per_target: Optional[Dict[str, List[str]]]=None
    select_features: Optional[List[str]] = None
    train_new_users_only: bool = False  # use only playerForTestSetAndFuturePreds=True for training
    on_season_only: bool = True  # use only on-season data for training
    gap_days: int = 0  # gap between end of training date vs start of validation date (days)
    train_days: int = 2000  # training period (days)
    model_type: str = 'lgb'
    train_full: bool = False
    upload: bool = False
    weight_new_users: bool = False
    extra_df_on: Optional[List[str]] = None
    drop_features: Optional[Union[List[str], Dict[str, List[str]]]] = None
    second_order_features: bool = False


@dataclass
class NNConfig:
    batch_size: int
    lr: float
    epochs: int
    batch_double_freq: int
    scaler_type: str
    optimizer: str
    weight_decay: float
    scheduler_type: str

    model_type: str

    max_lr: float

    emb_dim: int
    dropout_emb: float

    mlp_bn: bool
    mlp_dropout: float
    mlp_hidden: int

    cnn_hidden: int
    cnn_channel1: int
    cnn_channel2: int
    cnn_channel3: int
    cnn_dropout_top: float
    cnn_dropout_mid: float
    cnn_dropout_bottom: float
    cnn_weight_norm: bool
    cnn_two_stage: bool
    cnn_kernel1: int
    cnn_celu: bool

    seeds: List[int]

