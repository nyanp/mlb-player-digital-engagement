# import ujson
import json

import pandas as pd

from src.row import Row


def _to_datetime(d):
    if 'date' in d:
        date = d['date']
    else:
        date = d.name
    return pd.to_datetime(date, format='%Y%m%d').to_numpy()


def parse_row(row: pd.Series):
    def _parse(_row, idx):
        if idx in _row and not pd.isnull(_row[idx]):
            return json.loads(_row[idx])
        else:
            return []

    return Row(
        date=_to_datetime(row),
        engagement=_parse(row, 'nextDayPlayerEngagement'),
        games=_parse(row, 'games'),
        rosters=_parse(row, 'rosters'),
        player_box_scores=_parse(row, 'playerBoxScores'),
        team_box_scores=_parse(row, 'teamBoxScores'),
        transactions=_parse(row, 'transactions'),
        standings=_parse(row, 'standings'),
        awards=_parse(row, 'awards'),
        events=_parse(row, 'events'),
        player_twitter_followers=_parse(row, 'playerTwitterFollowers'),
        team_twitter_followers=_parse(row, 'teamTwitterFollowers')
    )


def make_df_base_from_test(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df['playerId'] = df['date_playerId'].apply(lambda x: int(x.split('_')[1]))
    df['dailyDataDate'] = _to_datetime(df)
    return df[['dailyDataDate', 'date', 'playerId', 'target1', 'target2', 'target3', 'target4']].set_index('date')


def make_df_base_from_train_engagement(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = df['dailyDataDate']
    df['dailyDataDate'] = _to_datetime(df)
    return df[['dailyDataDate', 'date', 'playerId', 'target1', 'target2', 'target3', 'target4']].set_index('date')
