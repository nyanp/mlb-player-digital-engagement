import gc
import os
import traceback
import warnings
from collections import defaultdict
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.event_level_model import prep_events
from src.parser import parse_row, Row
from src.player import Players
from src.streamdf import TimeSeriesStream
from src.team import Teams
from src.util import catch_everything_in_kaggle


class Store:
    def __init__(self,
                 players: Players,
                 teams: Teams,
                 event_model: Optional[List[lgb.Booster]] = None,
                 use_updated: bool = True,
                 debug: bool = False):
        self.players = players
        self.teams = teams
        self.daily_stats = TimeSeriesStream.empty({
            "daily_number_of_games": np.int32,
            "daily_number_of_awards": np.int32,
            "daily_number_of_transactions": np.int32,
            "daily_number_of_events": np.int32,
            'events_oof_target1_g_max': np.float32,
            'events_oof_target1_g_mean': np.float32,
            'events_oof_target2_g_max': np.float32,
            'events_oof_target2_g_mean': np.float32,
            'events_oof_target3_g_max': np.float32,
            'events_oof_target3_g_mean': np.float32,
            'events_oof_target4_g_max': np.float32,
            'events_oof_target4_g_mean': np.float32,
        }, 'dailyDataDate')
        self.game2gametype = {}  # gamePk => gameType
        self.gametype2games = defaultdict(list)
        self.last_timestamp = None
        self.event_model = event_model
        self.use_updated = use_updated
        self.debug = debug
        self.meta_dfs = []
        self.team_aggs = []
        self.daily_aggs = []

    @classmethod
    def empty(cls, data_dir: str, event_model: Optional[List[lgb.Booster]] = None, debug: bool = False) -> 'Store':
        player_df = pd.read_csv(os.path.join(data_dir, 'players.csv'))
        teams_df = pd.read_csv(os.path.join(data_dir, 'teams.csv'))

        teams = Teams(teams_df)
        players = Players(player_df, teams)

        return cls(players, teams, event_model=event_model, debug=debug)

    @classmethod
    def train(cls,
              data_dir: str,
              feature_dir: str = None,
              n: int = None,
              use_updated: bool = True,
              event_model: Optional[List[lgb.Booster]] = None,
              date_until: Optional[int] = None,
              debug: bool = False) -> 'Store':
        instance = cls.empty(data_dir, event_model=event_model, debug=debug)

        stem = 'train_updated' if use_updated else 'train'
        if feature_dir is not None and os.path.exists(os.path.join(feature_dir, f'{stem}.f')):
            df = pd.read_feather(os.path.join(feature_dir, f'{stem}.f'))
        elif os.path.exists(os.path.join(data_dir, f'{stem}.f')):
            df = pd.read_feather(os.path.join(data_dir, f'{stem}.f'))
        else:
            df = pd.read_csv(os.path.join(data_dir, f'{stem}.csv'))

        # 2018より前のデータを先に足す。
        awards = pd.read_csv(os.path.join(data_dir, 'awards.csv'))
        awards = awards.groupby('awardDate').apply(lambda x: x.to_json(orient='records')).reset_index()
        awards.columns = ['awardDate', 'awards']
        awards['date'] = awards['awardDate'].str.replace('-', '')
        awards = awards.sort_values(by='date', ascending=True).reset_index(drop=True)
        for i, row in tqdm(awards.iterrows()):
            instance.append(row.copy())

        df = df.set_index('date')

        for i, row in tqdm(df.iterrows()):
            if date_until is not None and row.name > date_until:
                break
            instance.append(row.copy())
            if n is not None and i >= n:
                break
            if i % 100 == 0:
                gc.collect()

        del df
        gc.collect()

        instance.use_updated = use_updated

        if not use_updated:
            warnings.warn("This store is using pre-updated train.csv!!!!!!!!")

        return instance

    def make_daily_stats(self, row: Row) -> Dict:
        return {
            "daily_number_of_games": len(row.games),
            "daily_number_of_awards": len(row.awards),
            "daily_number_of_transactions": len(row.transactions),
            "daily_number_of_events": len(row.events)
        }

    def calculate_event_meta_prediction(self, row: Row):
        df = pd.DataFrame(row.events)
        event_df = prep_events(df, sort_by_date=False)
        x = event_df.drop(['playerId', 'teamId'], axis=1).astype(np.float32)

        for i in range(4):
            feature_name = self.event_model[i].feature_name()
            event_df[f'predicted{i + 1}'] = self.event_model[i].predict(x[feature_name])

        if self.debug:
            self.meta_dfs.append(event_df.copy())

        aggregated = event_df.groupby(['playerId'])[[f'predicted{i + 1}' for i in range(4)]].agg(
            ['min', 'max', 'mean']).reset_index()

        cols = ['playerId']
        for i in range(4):
            tgt = f"target{i + 1}"
            cols += [f'events_oof_{tgt}_min', f'events_oof_{tgt}_max', f'events_oof_{tgt}_mean']
        aggregated.columns = cols

        daily_agg = {
            'events_oof_target1_g_max': event_df['predicted1'].max(),
            'events_oof_target1_g_mean': event_df['predicted1'].mean(),
            'events_oof_target2_g_max': event_df['predicted2'].max(),
            'events_oof_target2_g_mean': event_df['predicted2'].mean(),
            'events_oof_target3_g_max': event_df['predicted3'].max(),
            'events_oof_target3_g_mean': event_df['predicted3'].mean(),
            'events_oof_target4_g_max': event_df['predicted4'].max(),
            'events_oof_target4_g_mean': event_df['predicted4'].mean(),

        }

        team_agg = event_df.groupby(['teamId'])[[f'predicted{i + 1}' for i in range(4)]].agg(['max', 'mean']).reset_index()
        cols = ['teamId']
        for i in range(4):
            tgt = f"target{i + 1}"
            cols += [f'events_oof_{tgt}_t_max', f'events_oof_{tgt}_t_mean']
        team_agg.columns = cols

        if self.debug:
            self.team_aggs.append(team_agg)
            self.daily_aggs.append(daily_agg)

        return aggregated, daily_agg, team_agg

    def append(self, daily_data: pd.Series):
        row = parse_row(daily_data)

        with catch_everything_in_kaggle():
            self.teams.extend(row)

        with catch_everything_in_kaggle():
            self.players.extend(row)

        with catch_everything_in_kaggle():
            for game in row.games:
                self.game2gametype[game['gamePk']] = game['gameType']
                self.gametype2games[game['gameType']].append(game['gamePk'])

        for e in row.engagement:
            tid = self.players[e['playerId']].team_id
            if tid is not None:
                self.teams[tid].engagement.extend(e, row.date)

        with catch_everything_in_kaggle():
            daily_stats = self.make_daily_stats(row)

        if len(row.events) and self.event_model is not None:
            with catch_everything_in_kaggle():
                meta, daily_meta, team_meta = self.calculate_event_meta_prediction(row)
                self.players.extend_meta(meta, row.date)
                self.teams.extend_meta(team_meta, row.date)
                daily_stats.update(daily_meta)
        else:
            daily_stats['events_oof_target1_g_max'] = None
            daily_stats['events_oof_target1_g_mean'] = None
            daily_stats['events_oof_target2_g_max'] = None
            daily_stats['events_oof_target2_g_mean'] = None
            daily_stats['events_oof_target3_g_max'] = None
            daily_stats['events_oof_target3_g_mean'] = None
            daily_stats['events_oof_target4_g_max'] = None
            daily_stats['events_oof_target4_g_mean'] = None

        with catch_everything_in_kaggle():
            self.daily_stats.extend(daily_stats, row.date)

        self.last_timestamp = row.date

    def rollback(self, until: np.datetime64):
        self.players.rollback(until)
        self.teams.rollback(until)
        self.daily_stats = self.daily_stats.slice_until(until)
        self.last_timestamp = until
