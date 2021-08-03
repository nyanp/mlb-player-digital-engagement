from typing import Optional

import numpy as np
import pandas as pd
from IPython.display import display

from src.constants import is_on_season
from src.row import Row
from src.streamdf import TimeSeriesStream
from src.team import Teams, Team
from src.util import catch_everything_in_kaggle


PLAYER_POSITION_FEATURES = [
    'positionCode',
    'positionName',
    'positionType',
    'battingOrder'
]

PLAYER_BOX_FEATURES_2 = [
    'doubles',
    'triples',
    'groundIntoTriplePlay',
    'plateAppearances',
    'totalBases',
    'catchersInterference',
    'pickoffs',
    'flyOutsPitching',
    'airOutsPitching',
    'groundOutsPitching',
    'runsPitching',
    'doublesPitching',
    'triplesPitching',
    'homeRunsPitching',
    'strikeOutsPitching',
    'baseOnBallsPitching',
    'intentionalWalksPitching',
    'caughtStealingPitching',
    'stolenBasesPitching',
    'inningsPitched',
    'saveOpportunities',
    'earnedRuns',
    'battersFaced',
    'outsPitching',
    'pitchesThrown',
    'balls',
    'strikes',
    'hitBatsmen',
    'balks',
    'pickoffsPitching',
    'rbiPitching',
    'inheritedRunners',
    'inheritedRunnersScored',
    'catchersInterferencePitching',
    'sacBuntsPitching',
    'sacFliesPitching',
]

PLAYER_BOX_FEATUERS = [
    # 'battingOrder',
    'gamesPlayedBatting',
    'flyOuts',
    'groundOuts',
    'runsScored',
    'homeRuns',
    'strikeOuts',
    'baseOnBalls',
    'intentionalWalks',
    'hits',
    'hitByPitch',
    'atBats',
    'caughtStealing',
    'stolenBases',
    'rbi',
    'leftOnBase',
    'sacBunts',
    'sacFlies',
    'groundIntoDoublePlay',
    'gamesPlayedPitching',
    'gamesStartedPitching',
    'completeGamesPitching',
    'shutoutsPitching',
    'winsPitching',
    'lossesPitching',
    'hitsPitching',
    'hitByPitchPitching',
    'atBatsPitching',
    'wildPitches',
    'saves',
    'holds',
    'blownSaves',
    'assists',
    'putOuts',
    'errors',
    'chances'
]


class Player:
    def __init__(self,
                 row: pd.Series,
                 player_id: int,
                 teams: Teams,
                 engagement: TimeSeriesStream,
                 box_scores: TimeSeriesStream,
                 rosters: TimeSeriesStream,
                 transactions: TimeSeriesStream,
                 followers: TimeSeriesStream,
                 awards: TimeSeriesStream,
                 events: TimeSeriesStream,
                 events_meta: TimeSeriesStream,
                 last_timestamp: np.datetime64 = None
                 ):
        self.row = row
        self.player_id = player_id
        self.teams = teams
        self.last_timestamp = last_timestamp
        self.engagement = engagement
        self.box_scores = box_scores
        self.rosters = rosters
        self.transactions = transactions
        self.followers = followers
        self.awards = awards
        self.events = events
        self.events_meta = events_meta
        self._team = None

    @classmethod
    def empty(cls, row: pd.Series, player_id: int, teams: Optional[Teams] = None) -> 'Player':
        # nextDayPlayerEngagement
        engagement = TimeSeriesStream.empty({
            'target1': np.float32,
            'target2': np.float32,
            'target3': np.float32,
            'target4': np.float32,
            'on_season': bool
        }, 'dailyDataDate')

        # playerBoxScores
        box_score_schema = {
            'home': np.float32,
            'gamePk': np.float32,
            'gameDate': 'datetime64[D]',
            'teamId': np.float32,
            'teamName': object,
            'positionCode': object
        }
        for c in PLAYER_BOX_FEATUERS:
            box_score_schema[c] = np.float32
        for c in PLAYER_BOX_FEATURES_2:
            box_score_schema[c] = np.float32
        for c in PLAYER_POSITION_FEATURES:
            box_score_schema[c] = object
        box_scores = TimeSeriesStream.empty(box_score_schema, 'dailyDataDate')

        # rosters
        rosters = TimeSeriesStream.empty({
            'playerId': np.float32,
            'gameDate': 'datetime64[D]',
            'teamId': np.float32,
            'statusCode': object
        }, 'dailyDataDate')

        # transactions
        transactions = TimeSeriesStream.empty({
            'transactionId': np.float32,
            # 'playerId': np.float32,
            # 'date': 'datetime64[D]',
            'fromTeamId': np.float32,
            'toTeamId': np.float32,
            'typeCode': object,
            'resolutionDate': 'datetime64[D]',
            'effectiveDate': 'datetime64[D]',
        }, 'dailyDataDate')

        followers = TimeSeriesStream.empty({
            # 'date': 'datetime64[D]',
            # 'playerId': np.float32,
            'playerName': object,
            'accountName': object,
            'twitterHandle': object,
            'numberOfFollowers': np.float32
        }, 'dailyDataDate')

        awards = TimeSeriesStream.empty({
            'awardId': object,
            'awardDate': 'datetime64[D]'
        }, 'dailyDataDate')

        events = TimeSeriesStream.empty({
            'nastyFactor': np.float32,
            'type': object,
            'x0': np.float32,
            'startSpeed': np.float32,
            'totalDistance': np.float32,
            'launchSpeed': np.float32
        }, 'dailyDataDate')

        events_meta = TimeSeriesStream.empty({
            'events_oof_target1_min': np.float32,
            'events_oof_target1_max': np.float32,
            'events_oof_target1_mean': np.float32,
            'events_oof_target2_min': np.float32,
            'events_oof_target2_max': np.float32,
            'events_oof_target2_mean': np.float32,
            'events_oof_target3_min': np.float32,
            'events_oof_target3_max': np.float32,
            'events_oof_target3_mean': np.float32,
            'events_oof_target4_min': np.float32,
            'events_oof_target4_max': np.float32,
            'events_oof_target4_mean': np.float32,
        }, 'dailyDataDate')

        return Player(row, player_id, teams, engagement,
                      box_scores, rosters, transactions,
                      followers, awards, events, events_meta)

    @property
    def team(self) -> Optional[Team]:
        """
        現在所属しているチームの、今のタイムスタンプにおける情報を返す
        :return:
        """
        if self._team is not None:
            return self._team

        team_id = self.team_id
        if pd.isnull(team_id):
            return None
        team = self.teams[int(team_id)]
        if self.last_timestamp is not None:
            self._team = team.slice_until(self.last_timestamp)
            return self._team
        else:
            return team

    @property
    def team_id(self) -> Optional[int]:
        if self.teams is None:
            return None
        return self.rosters.last_value('teamId')

    def display(self):
        print(f'player-id: {self.player_id}')

        if self.row is not None:
            display(self.row)

        display(self.box_scores.to_df)
        display(self.rosters.to_df)
        display(self.transactions.to_df)

    def set_timestamp(self, ts: np.datetime64):
        if self.last_timestamp is None or ts > self.last_timestamp:
            self.last_timestamp = ts
        self._team = None

    def slice_until(self, until: np.datetime64) -> 'Player':
        # ts <= untilまでのデータでスライスする.
        return Player(self.row, self.player_id, self.teams,
                      self.engagement.slice_until(until), self.box_scores.slice_until(until),
                      self.rosters.slice_until(until), self.transactions.slice_until(until),
                      self.followers.slice_until(until),
                      self.awards.slice_until(until),
                      self.events.slice_until(until),
                      events_meta=self.events_meta.slice_until(until),
                      last_timestamp=until)


class Players:
    def __init__(self, player_df: pd.DataFrame, teams: Teams):
        self.player_df = player_df.set_index('playerId')
        self.players = {}  # type: Dict[int, Player]
        self.teams = teams

    def __getitem__(self, player_id: int) -> Player:
        if player_id not in self.players:
            self.players[player_id] = Player.empty(
                self.player_df.loc[player_id] if player_id in self.player_df.index else None,
                player_id, self.teams)
        return self.players[player_id]

    def extend_meta(self, meta: pd.DataFrame, date: np.datetime64):
        for i, row in meta.iterrows():
            self[row['playerId']].events_meta.extend(row.to_dict(), date)

    def extend(self, row: Row):
        updated_players = set()

        on_season = is_on_season(row.date)

        for b in row.player_box_scores:
            with catch_everything_in_kaggle():
                self[b['playerId']].box_scores.extend(b, row.date)
                updated_players.add(b['playerId'])

        for b in row.transactions:
            with catch_everything_in_kaggle():
                self[b['playerId']].transactions.extend(b, row.date)
                updated_players.add(b['playerId'])

        for b in row.rosters:
            with catch_everything_in_kaggle():
                self[b['playerId']].rosters.extend(b, row.date)
                updated_players.add(b['playerId'])

        for b in row.player_twitter_followers:
            with catch_everything_in_kaggle():
                self[b['playerId']].followers.extend(b, row.date)
                updated_players.add(b['playerId'])

        for b in row.awards:
            with catch_everything_in_kaggle():
                self[b['playerId']].awards.extend(b, row.date)
                updated_players.add(b['playerId'])

        for b in row.events:
            with catch_everything_in_kaggle():
                self[b['pitcherId']].events.extend(b, row.date)
                self[b['hitterId']].events.extend(b, row.date)

        if row.engagement:
            for b in row.engagement:
                b['on_season'] = on_season
                self[b['playerId']].engagement.extend(b, row.date)
                updated_players.add(b['playerId'])

        for playerId in updated_players:
            with catch_everything_in_kaggle():
                self[playerId].set_timestamp(row.date)

    def rollback(self, until: np.datetime64):
        self.players = {k: v.slice_until(until) for k, v in self.players.items()}
        if self.teams:
            self.teams = self.teams.rollback(until)
        return self
