from typing import Any, Dict
import numpy as np
import pandas as pd

from src.streamdf import TimeSeriesStream
from src.row import Row
from src.util import catch_everything_in_kaggle


TEAM_BOX_FEATURES = [
    'flyOuts',
    'groundOuts',
    'caughtStealing',
    'stolenBases',
    'groundIntoDoublePlay',
    #'groundIntoTriplePlay',
    'plateAppearances',
    'totalBases',
    'rbi',
    'leftOnBase',
    'sacBunts',
    'sacFlies',
    'catchersInterference',
    'pickoffs',
    #'airOutsPitching',
    #'groundOutsPitching',
    'runsPitching',
    #'doublesPitching',
    #'triplesPitching',
    'homeRunsPitching',
    'strikeOutsPitching',
    'baseOnBallsPitching',
    'intentionalWalksPitching',
    'hitsPitching',
    'hitByPitchPitching',
    'atBatsPitching',
    'caughtStealingPitching',
    'stolenBasesPitching',
    'inningsPitched',
    'earnedRuns',
    'battersFaced',
    'outsPitching',
    'hitBatsmen',
    #'balks',
    #'wildPitches',
    'pickoffsPitching',
    'rbiPitching',
    #'inheritedRunners',
    #'inheritedRunnersScored',
    'catchersInterferencePitching',
    'sacBuntsPitching',
    'sacFliesPitching'
]


class Team:
    def __init__(self, row: pd.Series, team_id: int,
                 engagement: TimeSeriesStream,
                 box_scores: TimeSeriesStream,
                 followers: TimeSeriesStream,
                 standings: TimeSeriesStream,
                 games: TimeSeriesStream,
                 events_meta: TimeSeriesStream):
        self.row = row
        self.team_id = team_id
        self.engagement = engagement
        self.box_scores = box_scores
        self.followers = followers
        self.standings = standings
        self.games = games
        self.events_meta = events_meta

    @classmethod
    def empty(cls, row: pd.Series, team_id: int):
        box_score_schema = {
            'home': np.float32,
            'gamePk': np.float32,
            # 'gameDate': 'datetime64[D]',
            'gameTimeUTC': 'datetime64[s]'
        }
        for c in TEAM_BOX_FEATURES:
            box_score_schema[c] = np.float32
        box_scores = TimeSeriesStream.empty(box_score_schema, 'dailyDataDate')

        engagement = TimeSeriesStream.empty({
            'playerId': np.int32,
            'target1': np.float32,
            'target2': np.float32,
            'target3': np.float32,
            'target4': np.float32
        }, 'dailyDataDate')

        followers = TimeSeriesStream.empty({
            # 'date': 'datetime64[D]',
            'accountName': object,
            'twitterHandle': object,
            'numberOfFollowers': np.float32
        }, 'dailyDataDate')

        standings = TimeSeriesStream.empty({
            'divisionId': np.float32,
            'streakCode': object,
            'divisionRank': np.float32,
            'leagueRank': np.float32,
            'wildCardRank': np.float32,
            'leagueGamesBack': object,
            'sportGamesBack': object,
            'divisionGamesBack': object,
            'wins': np.float32,
            'losses': np.float32,
            'pct': np.float32,
            'runsAllowed': np.float32,
            'runsScored': np.float32,
            'divisionChamp': np.float32,
            'divisionLeader': np.float32,
            'wildCardLeader': object,
            'eliminationNumber': object,
            'wildCardEliminationNumber': object,
            'homeWins': np.float32,
            'homeLosses': np.float32,
            'awayWins': np.float32,
            'awayLosses': np.float32,
            'lastTenWins': np.float32,
            'lastTenLosses': np.float32,
            'extraInningWins': np.float32,
            'extraInningLosses': np.float32,
            'oneRunWins': np.float32,
            'oneRunLosses': np.float32,
            'dayWins': np.float32,
            'dayLosses': np.float32,
            'nightWins': np.float32,
            'nightLosses': np.float32,
            'grassWins': np.float32,
            'grassLosses': np.float32,
            'turfWins': np.float32,
            'turfLosses': np.float32,
            'divWins': np.float32,
            'divLosses': np.float32,
            'alWins': np.float32,
            'alLosses': np.float32,
            'nlWins': np.float32,
            'nlLosses': np.float32,
            'xWinLossPct': np.float32,
        }, 'dailyDataDate')

        games = TimeSeriesStream.empty({
            'gamePk': np.float32,
            'gameTimeUTC': object,
            'gameType': object,
            'dayNight': object,
            'teamWins': np.float32,
            'teamWinPct': np.float32,
            'teamWinner': np.float32,
            'teamScore': np.float32,
            'opponentTeamScore': np.float32,
            'isTie': np.float32,
            #'doubleHeader': object,
            'is_home_game': np.int32
        }, 'dailyDataDate')

        events_meta = TimeSeriesStream.empty({
            'events_oof_target1_t_max': np.float32,
            'events_oof_target1_t_mean': np.float32,
            'events_oof_target2_t_max': np.float32,
            'events_oof_target2_t_mean': np.float32,
            'events_oof_target3_t_max': np.float32,
            'events_oof_target3_t_mean': np.float32,
            'events_oof_target4_t_max': np.float32,
            'events_oof_target4_t_mean': np.float32,
        }, 'dailyDataDate')

        return Team(row, team_id,
                    engagement, box_scores,
                    followers, standings, games, events_meta)

    def slice_until(self, until: np.datetime64) -> 'Team':
        return Team(self.row, self.team_id,
                    self.engagement.slice_until(until),
                    self.box_scores.slice_until(until),
                    self.followers.slice_until(until), self.standings.slice_until(until),
                    self.games.slice_until(until),
                    self.events_meta.slice_until(until))


class Teams:
    def __init__(self, team_df: pd.DataFrame):
        self.team_df = team_df.set_index('id')
        self.teams = {}  # type: Dict[int, Team]

    def __getitem__(self, team_id: int) -> Team:
        if team_id not in self.teams:
            self.teams[team_id] = Team.empty(self.team_df.loc[team_id] if team_id in self.team_df.index else None, team_id)
        return self.teams[team_id]

    def extend(self, row: Row):
        for b in row.team_box_scores:
            with catch_everything_in_kaggle():
                self[b['teamId']].box_scores.extend(b, row.date)

        for b in row.team_twitter_followers:
            with catch_everything_in_kaggle():
                self[b['teamId']].followers.extend(b, row.date)

        for b in row.standings:
            with catch_everything_in_kaggle():
                self[b['teamId']].standings.extend(b, row.date)

        for g in row.games:
            with catch_everything_in_kaggle():
                g_home = normalize_game(g, 'home')
                g_away = normalize_game(g, 'away')

                self[g_home['teamId']].games.extend(g_home, row.date)
                self[g_away['teamId']].games.extend(g_away, row.date)

    def extend_meta(self, meta: pd.DataFrame, date: np.datetime64):
        for i, row in meta.iterrows():
            with catch_everything_in_kaggle():
                self[row['teamId']].events_meta.extend(row.to_dict(), date)

    def rollback(self, until: np.datetime64):
        self.teams = {k: v.slice_until(until) for k, v in self.teams.items()}
        return self


def normalize_game(game: Dict[str, Any], home_or_away: str = 'home'):
    col_map = {}

    if home_or_away == 'home':
        prefix_to_home = 'team'
        prefix_to_away = 'opponentTeam'
    else:
        prefix_to_away = 'team'
        prefix_to_home = 'opponentTeam'

    for key in game.keys():
        if key.startswith('home'):
            col_map[key] = key.replace('home', prefix_to_home)
        elif key.startswith('away'):
            col_map[key] = key.replace('away', prefix_to_away)

    d = {col_map.get(k, k): v for k, v in game.items()}
    d['is_home_game'] = 1 if home_or_away == 'home' else 0
    return d
