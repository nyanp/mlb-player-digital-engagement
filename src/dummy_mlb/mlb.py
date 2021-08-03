import json
import os
import random
import warnings
from typing import Optional, Tuple
from sklearn.metrics import mean_absolute_error

import numpy as np
import pandas as pd

from src.parser import make_df_base_from_train_engagement, make_df_base_from_test


class Environment:
    def __init__(self,
                 data_dir: str,
                 eval_start_day: int,
                 eval_end_day: Optional[int],
                 use_updated: bool,
                 multiple_days_per_iter: bool,
                 chaos_level: int = 0,
                 chaos_probability: float = 0.5):
        warnings.warn('this is mock module for mlb')

        postfix = '_updated' if use_updated else ''
        try:
            df_train = pd.read_feather(os.path.join(data_dir, f'train{postfix}.f'))
        except Exception:
            df_train = pd.read_csv(os.path.join(data_dir, f'train{postfix}.csv'))
        players = pd.read_csv(os.path.join(data_dir, 'players.csv'))

        self.players = players[players['playerForTestSetAndFuturePreds'] == True]['playerId'].astype(str)
        if eval_end_day is not None:
            self.df_train = df_train.set_index('date').loc[eval_start_day:eval_end_day]
        else:
            self.df_train = df_train.set_index('date').loc[eval_start_day:]
        self.date = self.df_train.index.values
        self.n_rows = len(self.df_train)
        self.multiple_days_per_iter = multiple_days_per_iter
        self.predicted = []
        self.chaos_level = chaos_level
        self.chaos_probability = chaos_probability

        assert self.n_rows > 0, 'no data to emulate'

    def predict(self, df: pd.DataFrame) -> None:
        self.predicted.append(df)

    def _chaos_monkey(self, df_train: pd.DataFrame, df_sub: pd.DataFrame):
        if self.chaos_level >= 1:
            # Lv1: randomly drop column
            if np.random.rand() < self.chaos_probability:
                drop_col = np.random.choice([
                    'games', 'rosters', 'playerBoxScores', 'teamBoxScores', 'transactions',
                    'standings', 'awards', 'events', 'playerTwitterFollowers',
                    'teamTwitterFollowers'])
                df_train[drop_col] = None

            # Lv1: random shuffle
            if np.random.rand() < self.chaos_probability:
                df_train = df_train.sample(len(df_train))
                df_sub = df_sub.sample(len(df_sub))

        if self.chaos_level >= 2:
            # Lv2: different column order
            if np.random.rand() < self.chaos_probability:
                cols = list(df_train.columns)
                random.shuffle(cols)
                df_train = df_train[cols]

            # Lv2: swap column schema
            if np.random.rand() < self.chaos_probability:
                src_col = np.random.choice([
                    'games', 'rosters', 'playerBoxScores', 'teamBoxScores', 'transactions',
                    'standings', 'awards', 'events', 'playerTwitterFollowers',
                    'teamTwitterFollowers'])
                dst_col = np.random.choice([
                    'games', 'rosters', 'playerBoxScores', 'teamBoxScores', 'transactions',
                    'standings', 'awards', 'events', 'playerTwitterFollowers',
                    'teamTwitterFollowers'])
                df_train[dst_col] = df_train[src_col]

            # Lv2: randomly drop multiple columns
            if np.random.rand() < self.chaos_probability:
                for i in range(np.random.randint(1, 4)):
                    drop_col = np.random.choice([
                        'games', 'rosters', 'playerBoxScores', 'teamBoxScores', 'transactions',
                        'standings', 'awards', 'events', 'playerTwitterFollowers',
                        'teamTwitterFollowers'])
                    df_train[drop_col] = None

            # Lv2: randomly change value type inside json
            if np.random.rand() < self.chaos_probability:
                drop_col = np.random.choice([
                    'games', 'rosters', 'playerBoxScores', 'teamBoxScores', 'transactions',
                    'standings', 'awards', 'events', 'playerTwitterFollowers',
                    'teamTwitterFollowers'])
                if df_train[drop_col].iloc[0] is not None:
                    json_data = json.loads(df_train[drop_col].iloc[0])
                    json_dst = []

                    def _change(val):
                        return val if np.random.rand() < 0.9 else 'foobar'

                    for row in json_data:
                        json_dst.append({k: _change(v) for k, v in row.items() if np.random.rand() < 0.2})
                    df_train[drop_col].iloc[0] = json.dumps(json_dst)

        if self.chaos_level >= 3:
            # Lv3: corrupt index
            if np.random.rand() < self.chaos_probability:
                df_train.index = ['A']*len(df_train)

            # Lv3: randomly drop fields inside json
            if np.random.rand() < self.chaos_probability:
                drop_col = np.random.choice([
                    'games', 'rosters', 'playerBoxScores', 'teamBoxScores', 'transactions',
                    'standings', 'awards', 'events', 'playerTwitterFollowers',
                    'teamTwitterFollowers'])
                if df_train[drop_col].iloc[0] is not None:
                    json_data = json.loads(df_train[drop_col].iloc[0])
                    json_dst = []

                    for row in json_data:
                        json_dst.append({k: v for k, v in row.items() if np.random.rand() < 0.8})
                    df_train[drop_col].iloc[0] = json.dumps(json_dst)

        return df_train, df_sub

    def iter_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.multiple_days_per_iter:
            for i in range(self.n_rows // 2):
                date1 = self.date[2 * i]
                date2 = self.date[2 * i + 1]
                sample_sub1 = self._make_sample_sub(date1)
                sample_sub2 = self._make_sample_sub(date2)
                sample_sub = pd.concat([sample_sub1, sample_sub2]).reset_index(drop=True)
                df = self.df_train.loc[date1:date2]

                yield df.drop('nextDayPlayerEngagement', axis=1), sample_sub.set_index('date')
        else:
            for i in range(self.n_rows):
                date = self.date[i]
                print(f'date:{date}')
                sample_sub = self._make_sample_sub(date)
                df = self.df_train.loc[date:date]

                yield df.drop('nextDayPlayerEngagement', axis=1), sample_sub.set_index('date')

    def _make_sample_sub(self, date: int) -> pd.DataFrame:
        next_day = (pd.to_datetime(date, format='%Y%m%d') + pd.to_timedelta(1, 'd')).strftime('%Y%m%d')
        sample_sub = pd.DataFrame()
        sample_sub['date_playerId'] = next_day + '_' + self.players
        sample_sub['target1'] = 0
        sample_sub['target2'] = 0
        sample_sub['target3'] = 0
        sample_sub['target4'] = 0
        sample_sub['date'] = date
        return sample_sub

    def ground_truth_df(self) -> pd.DataFrame:
        eng = []

        for i, row in self.df_train.iterrows(): #['nextDayPlayerEngagement']:
            try:
                parsed = json.loads(row['nextDayPlayerEngagement'])
                for r in parsed:
                    r['dailyDataDate'] = row.name
                eng.extend(parsed)
            except:
                raise

        return make_df_base_from_train_engagement(pd.DataFrame(eng))

    def predicted_df(self) -> pd.DataFrame:
        return make_df_base_from_test(pd.concat(self.predicted))

    def mae(self) -> Tuple[float, float, float, float]:
        pred = self.predicted_df()
        gt = self.ground_truth_df()
        gt.columns = ['dailyDataDate', 'playerId', 'target1_gt', 'target2_gt', 'target3_gt', 'target4_gt']
        pred = pd.merge(pred, gt, on=['dailyDataDate', 'playerId'], how='left')

        return tuple([float(mean_absolute_error(pred[f"target{i+1}_gt"], pred[f"target{i+1}"])) for i in range(4)])


class MLBEmulator:
    def __init__(self,
                 data_dir: str = '../input/mlb-player-digital-engagement-forecasting',
                 eval_start_day: int = 20210401,
                 eval_end_day: Optional[int] = 20210430,
                 use_updated: bool = True,
                 multiple_days_per_iter: bool = False,
                 chaos_level: int = 0,
                 chaos_probability: float = 0.5
                 ):
        self.data_dir = data_dir
        self.eval_start_day = eval_start_day
        self.eval_end_day = eval_end_day
        self.use_updated = use_updated
        self.multiple_days_per_iter = multiple_days_per_iter
        self.chaos_level = chaos_level
        self.chaos_probability = chaos_probability

    def make_env(self) -> Environment:
        return Environment(self.data_dir,
                           self.eval_start_day,
                           self.eval_end_day,
                           self.use_updated,
                           self.multiple_days_per_iter,
                           self.chaos_level,
                           self.chaos_probability)
