# merge tables & obvious features
from typing import Dict

import numpy as np
import pandas as pd

from src.constants import LABELS
from src.features import *
from src.features.helper import diff_days, day_diff
from src.player import PLAYER_BOX_FEATUERS, PLAYER_BOX_FEATURES_2
from src.team import TEAM_BOX_FEATURES
from src.streamdf import TimeSeriesStream


@feature(['primaryPositionCode', 'birthCountry'])
def f000_player_info(ctx: Context) -> Dict:
    features = [
        "primaryPositionCode",
        "birthCountry"
    ]
    if ctx.player.row is None:
        return {f: None for f in features}
    return {f: ctx.player.row[f] for f in features}


@feature(['player_awards_diff_days'])
def f001_player_awards(ctx: Context) -> Dict:
    d = {
        'player_awards_diff_days': diff_days(ctx.daily_data_date, ctx.player.awards)
    }
    return d


@feature(['player_' + c for c in PLAYER_BOX_FEATUERS] + ['player_box_scores_diff_days'])
def f002_player_box_scores(ctx: Context) -> Dict:
    d = {'player_' + c: ctx.player.box_scores.last_value(c) for c in PLAYER_BOX_FEATUERS}
    d['player_box_scores_diff_days'] = diff_days(ctx.daily_data_date, ctx.player.box_scores)
    return d


@feature(['player_statusCode', 'player_rosters_diff_days'])
def f003_player_rosters(ctx: Context) -> Dict:
    d = {
        'player_statusCode': ctx.player.rosters.last_value('statusCode'),
        'player_rosters_diff_days': diff_days(ctx.daily_data_date, ctx.player.rosters)
    }
    return d


@feature(['player_transactions_typeCode', 'player_transactions_diff_days'])
def f004_player_transactions(ctx: Context) -> Dict:
    d = {
        'player_transactions_typeCode': ctx.player.transactions.last_value('typeCode'),
        'player_transactions_diff_days': diff_days(ctx.daily_data_date, ctx.player.transactions)
    }
    return d


@feature(['team_' + c for c in TEAM_BOX_FEATURES] + ['team_box_scores_diff_days'])
def f005_team_box_scores(ctx: Context) -> Dict:
    team = ctx.team
    if team is None:
        d = {'team_' + k: None for k in TEAM_BOX_FEATURES}
    else:
        d = {'team_' + c: team.box_scores.last_value(c) for c in TEAM_BOX_FEATURES}

    if team is not None:
        d['team_box_scores_diff_days'] = diff_days(ctx.daily_data_date, team.box_scores)
    else:
        d['team_box_scores_diff_days'] = None

    return d


@feature(['team_game_wins', 'team_game_winPct', 'team_game_winner', 'team_game_score', 'team_game_isTie',
          'team_game_gameType', 'team_game_dayNight', 'team_game_diff_days'])
def f006_team_games(ctx: Context) -> Dict:
    team = ctx.team
    if team is None or len(team.games) == 0:
        d = {
            'team_game_wins': None,
            'team_game_winPct': None,
            'team_game_winner': None,
            'team_game_score': None,
            'team_game_isTie': None,
            'team_game_gameType': None,
            'team_game_dayNight': None,
            'team_game_diff_days': None
        }
    else:
        d = {
            'team_game_wins': team.games['teamWins'][-1],
            'team_game_winPct': team.games['teamWinPct'][-1],
            'team_game_winner': team.games['teamWinner'][-1],
            'team_game_score': team.games['teamScore'][-1],
            'team_game_isTie': team.games['isTie'][-1],
            'team_game_gameType': team.games['gameType'][-1],
            'team_game_dayNight': team.games['dayNight'][-1],
            'team_game_diff_days': diff_days(ctx.daily_data_date, team.games)
        }
    return d


@feature(['player_numberOfFollowers', 'player_twitter_diff_days', 'player_twitter_count'])
def f007_player_twitter_followers(ctx: Context) -> Dict:
    feature = {
        'player_numberOfFollowers': ctx.player.followers.last_value('numberOfFollowers'),
        'player_twitter_diff_days': diff_days(ctx.daily_data_date, ctx.player.followers),
        'player_twitter_count': len(ctx.player.followers)
    }

    return feature


@feature(['is_home_game'])
def f008_team_games(ctx: Context) -> Dict:
    team = ctx.team
    if team is None or len(team.games) == 0:
        d = {
            'is_home_game': None
        }
    else:
        d = {
            'is_home_game': team.games['is_home_game'][-1]
        }
    return d


@feature(['positionCode', 'positionName', 'positionType'])
def f009_player_position(ctx: Context) -> Dict:
    d = {
        'positionCode': ctx.player.box_scores.last_value('positionCode'),
        'positionName': ctx.player.box_scores.last_value('positionName'),
        'positionType': ctx.player.box_scores.last_value('positionType')
    }
    return d


@feature(['positionCodeRatio', 'positionNameRatio'])
def f010_player_position_ratio(ctx: Context) -> Dict:
    if len(ctx.player.box_scores) == 0:
        return empty_feature(ctx.current_feature_name)

    def _ratio(col):
        last_value = ctx.player.box_scores.last_value(col)
        return (ctx.player.box_scores[col] == last_value).mean()

    d = {
        'positionCodeRatio': _ratio('positionCode'),
        'positionNameRatio': _ratio('positionName')
    }
    return d


@feature(['player_box_scores_diff_days2', 'player_box_scores_diff_days3'])
def f011_player_box_scores_days_diff(ctx: Context) -> Dict:
    d = {
        'player_box_scores_diff_days2': diff_days(ctx.daily_data_date, ctx.player.box_scores, -2),
        'player_box_scores_diff_days3': diff_days(ctx.daily_data_date, ctx.player.box_scores, -3)
    }
    return d


@feature(['team_box_scores_diff_days2', 'team_box_scores_diff_days3'])
def f012_team_box_scores_days_diff(ctx: Context) -> Dict:
    team = ctx.team
    if team is None:
        return empty_feature(ctx.current_feature_name)

    d = {
        'team_box_scores_diff_days2': diff_days(ctx.daily_data_date, team.box_scores, -2),
        'team_box_scores_diff_days3': diff_days(ctx.daily_data_date, team.box_scores, -3)
    }
    return d


@feature(['awards_count', 'transactions_count'])
def f014_awards_count(ctx: Context) -> Dict:
    d = {
        'awards_count': len(ctx.player.awards),
        'transactions_count': len(ctx.player.transactions),
    }
    return d


@feature(['divisionId', 'divisionRank', 'leagueRank', 'wildCardRank'])
def f015_standings(ctx: Context) -> Dict:
    if ctx.team is None:
        return empty_feature(ctx.current_feature_name)
    standings = ctx.team.standings
    d = {
        'divisionId': standings.last_value('divisionId'),
        'divisionRank': standings.last_value('divisionRank'),
        'leagueRank': standings.last_value('leagueRank'),
        'wildCardRank': standings.last_value('wildCardRank')
    }
    return d


@feature([
    'day_from_last_ASG', 'day_from_last_SFA', 'day_from_last_SC',
    'day_from_last_OPT', 'day_from_last_CU'])
def f016_transaction_count(ctx: Context) -> Dict:
    transactions = ctx.player.transactions
    if len(transactions) == 0:
        return empty_feature(ctx.current_feature_name)

    typecode = transactions['typeCode']
    dates = transactions.index

    def _diff(a, b):
        if len(a) == 0:
            return None
        return (b - a[-1]) / np.timedelta64(1, 'D')

    d = {
        'day_from_last_ASG': _diff(dates[typecode == 'ASG'], ctx.daily_data_date),
        'day_from_last_SFA': _diff(dates[typecode == 'SFA'], ctx.daily_data_date),
        'day_from_last_SC': _diff(dates[typecode == 'SC'], ctx.daily_data_date),
        'day_from_last_OPT': _diff(dates[typecode == 'OPT'], ctx.daily_data_date),
        'day_from_last_CU': _diff(dates[typecode == 'CU'], ctx.daily_data_date)
    }
    return d


@feature([
    'latest_events_count', 'latest_events_nastyFactor_mean',
    'latest_events_startSpeed_max', 'latest_events_totalDistance_mean',
    'latest_events_totalDistance_max'
])
def f017_events(ctx: Context) -> Dict:
    if len(ctx.player.events) == 0:
        return empty_feature(ctx.current_feature_name)
    last_ts = ctx.player.events.index[-1]
    events_on_latest_game = ctx.player.events.slice_from(last_ts)

    d = {
        'latest_events_count': len(events_on_latest_game),
        'latest_events_nastyFactor_mean': events_on_latest_game.nanmean('nastyFactor'),
        'latest_events_startSpeed_max': events_on_latest_game.nanmax('startSpeed'),
        'latest_events_totalDistance_mean': events_on_latest_game.nanmean('totalDistance'),
        'latest_events_totalDistance_max': events_on_latest_game.nanmax('totalDistance')
    }
    return d


@feature(['player_statusCode_changed', 'player_team_changed'])
def f018_player_rosters_changed(ctx: Context) -> Dict:
    if len(ctx.player.rosters) < 2:
        return empty_feature(ctx.current_feature_name)

    rosters = ctx.player.rosters

    d = {
        'player_statusCode_changed': int(rosters['statusCode'][-1] != rosters['statusCode'][-2]),
        'player_team_changed': int(rosters['teamId'][-1] != rosters['teamId'][-2])
    }
    return d


@feature([
    'latest_events_launchSpeed_max',
    'latest_events_isGB_sum', 'latest_events_isLD_sum',
    'latest_events_isFB_sum', 'latest_events_isPU_sum'
])
def f019_events(ctx: Context) -> Dict:
    if len(ctx.player.events) == 0:
        return empty_feature(ctx.current_feature_name)
    last_ts = ctx.player.events.index[-1]
    events_on_latest_game = ctx.player.events.slice_from(last_ts)

    d = {
        'latest_events_launchSpeed_max': events_on_latest_game.nanmax('launchSpeed'),
        'latest_events_isGB_sum': events_on_latest_game.nansum('isGB'),
        'latest_events_isLD_sum': events_on_latest_game.nansum('isLD'),
        'latest_events_isFB_sum': events_on_latest_game.nansum('isFB'),
        'latest_events_isPU_sum': events_on_latest_game.nansum('isPU')
    }
    return d


@feature([
    'latest_events_action_count', 'latest_events_pitch_count'
])
def f020_events(ctx: Context) -> Dict:
    if len(ctx.player.events) == 0:
        return empty_feature(ctx.current_feature_name)
    last_ts = ctx.player.events.index[-1]
    events_on_latest_game = ctx.player.events.slice_from(last_ts)

    mask = events_on_latest_game['type'] == 'action'

    d = {
        'latest_events_action_count': int(mask.sum()),
        'latest_events_pitch_count': int((~mask).sum())
    }
    return d


@feature(['team_game_wins', 'team_game_winPct', 'team_game_winner', 'team_game_score', 'team_game_isTie',
          'team_game_gameType', 'team_game_dayNight', 'team_game_diff_days'])
def f021_team_games_2(ctx: Context) -> Dict:
    team = ctx.team

    if team is None or len(team.games) == 0:
        d = {
            'team_game_wins': None,
            'team_game_winPct': None,
            'team_game_winner': None,
            'team_game_score': None,
            'team_game_isTie': None,
            'team_game_gameType': None,
            'team_game_dayNight': None,
            'team_game_diff_days': None
        }
    else:
        d = {
            'team_game_wins': team.games['teamWins'][-1],
            'team_game_winPct': team.games['teamWinPct'][-1],
            'team_game_winner': team.games['teamWinner'][-1],
            'team_game_score': team.games['teamScore'][-1],
            'team_game_isTie': float(team.games['isTie'][-1]),
            'team_game_gameType': LABELS['gameType'].get(team.games['gameType'][-1], 6),
            'team_game_dayNight': int(team.games['dayNight'][-1] == 'day'),
            'team_game_diff_days': diff_days(ctx.daily_data_date, team.games)
        }
    return d


@feature(['primaryPositionCode', 'birthCountry'])
def f022_player_info_2(ctx: Context) -> Dict:
    if ctx.player.row is None:
        return empty_feature(ctx.current_feature_name)
    return {
        'primaryPositionCode': LABELS['positionCode'].get(ctx.player.row['primaryPositionCode'], 10),
        'birthCountry': LABELS['birthCountry'].get(ctx.player.row['birthCountry'], 10)
    }


@feature(['player_transactions_typeCode', 'player_transactions_diff_days'])
def f023_player_transactions_2(ctx: Context) -> Dict:
    d = {
        'player_transactions_typeCode': LABELS['typeCode'].get(ctx.player.transactions.last_value('typeCode'), 16),
        'player_transactions_diff_days': diff_days(ctx.daily_data_date, ctx.player.transactions)
    }
    return d


@feature(['player_statusCode', 'player_rosters_diff_days'])
def f024_player_rosters_2(ctx: Context) -> Dict:
    d = {
        'player_statusCode': LABELS['statusCode'].get(ctx.player.rosters.last_value('statusCode'), 10),
        'player_rosters_diff_days': diff_days(ctx.daily_data_date, ctx.player.rosters)
    }
    return d


@feature(['team_game_scorediff'])
def f025_team_games(ctx: Context) -> Dict:
    team = ctx.team
    if team is None or len(team.games) == 0:
        d = {
            'team_game_scorediff': None
        }
    else:
        d = {
            'team_game_scorediff': float(team.games['teamScore'][-1] - team.games['opponentTeamScore'][-1])
        }
    return d


@feature(['team_game_dt'])
def f026_team_games_dt(ctx: Context) -> Dict:
    team = ctx.team
    if team is None or len(team.games) == 0:
        return empty_feature(ctx.current_feature_name)

    try:
        last_game_utc = np.datetime64(team.games['gameTimeUTC'][-1][:-1])

        return {
            'team_game_dt': (ctx.daily_data_date - last_game_utc) / np.timedelta64(1, 'D')
        }
    except Exception:
        return empty_feature(ctx.current_feature_name)


@feature(['player_days_from_debut'])
def f027_days_from_debut(ctx: Context) -> Dict:
    try:
        debut_day = np.datetime64(ctx.player.row['mlbDebutDate'])

        return {
            'player_days_from_debut': (ctx.daily_data_date - debut_day) / np.timedelta64(1, 'D')
        }
    except Exception:
        return empty_feature(ctx.current_feature_name)


@feature(['player_' + c for c in PLAYER_BOX_FEATURES_2])
def f028_player_box_scores(ctx: Context) -> Dict:
    d = {'player_' + c: ctx.player.box_scores.last_value(c) for c in PLAYER_BOX_FEATURES_2}
    return d


@feature(['days_from_allstar'])
def f029_days_from_allstar_game(ctx: Context) -> Dict:
    if len(ctx.player.box_scores) == 0:
        return empty_feature(ctx.current_feature_name)

    allstar_or_ws = ctx.store.gametype2games['A'] + ctx.store.gametype2games['W']
    mask = np.in1d(ctx.player.box_scores['gamePk'], allstar_or_ws)

    if mask.sum() == 0:
        return empty_feature(ctx.current_feature_name)

    last_game = ctx.player.box_scores.index[mask][-1]

    return {
        'days_from_allstar': (ctx.daily_data_date - last_game) / np.timedelta64(1, 'D')
    }


@feature(['lastTenWins', 'pct', 'xWinLossPct'])
def f030_standings_2(ctx: Context) -> Dict:
    if ctx.team is None:
        return empty_feature(ctx.current_feature_name)
    standings = ctx.team.standings
    d = {
        'lastTenWins': standings.last_value('lastTenWins'),
        'pct': standings.last_value('pct'),
        'xWinLossPct': standings.last_value('xWinLossPct')
    }
    return d


@feature(['days_from_team_changed'])
def f031_days_from_team_changed(ctx: Context) -> Dict:
    trn = ctx.player.transactions
    if len(trn) == 0:
        return empty_feature(ctx.current_feature_name)

    team_changed = (trn['fromTeamId'] != trn['toTeamId']) & ~pd.isnull(trn['fromTeamId'])

    changed = trn.index[team_changed]

    if len(changed) == 0:
        return empty_feature(ctx.current_feature_name)

    d = {
        'days_from_team_changed': (ctx.daily_data_date - changed[-1]) / np.timedelta64(1, 'D')
    }
    return d


@feature(['award_id'])
def f032_award_id(ctx: Context) -> Dict:
    awards = ctx.player.awards
    if len(awards) == 0:
        return empty_feature(ctx.current_feature_name)

    return {
        'award_id': LABELS['award'].get(awards.last_value('awardId'), 100)
    }


@feature(['player_transactions_resolutionDate_diff', 'player_transactions_effectiveDate_diff'])
def f033_player_transactions(ctx: Context) -> Dict:
    dd = ctx.daily_data_date
    trn = ctx.player.transactions
    d = {
        'player_transactions_resolutionDate_diff': day_diff(dd, trn.last_value('resolutionDate')),
        'player_transactions_effectiveDate_diff': day_diff(dd, trn.last_value('effectiveDate'))
    }
    return d


@feature(['positionName', 'positionType'])
def f034_player_position(ctx: Context) -> Dict:
    d = {
        'positionName': LABELS['positionName'].get(ctx.player.box_scores.last_value('positionName'), 11),
        'positionType': LABELS['positionType'].get(ctx.player.box_scores.last_value('positionType'), 7)
    }
    return d


@feature(['awards_count_70days'])
def f035_awards_count_70days(ctx: Context) -> Dict:
    mask = ctx.player.awards.index >= ctx.daily_data_date - np.timedelta64(70, 'D')
    d = {
        'awards_count_70days': mask.sum()
    }
    return d


@feature(['awards_count_140days'])
def f036_awards_count_140days(ctx: Context) -> Dict:
    mask = ctx.player.awards.index >= ctx.daily_data_date - np.timedelta64(140, 'D')
    d = {
        'awards_count_140days': mask.sum()
    }
    return d


@feature([
    'latest_events_count'
])
def f037_events(ctx: Context) -> Dict:
    if len(ctx.player.events) == 0:
        return empty_feature(ctx.current_feature_name)
    last_ts = ctx.player.events.index[-1]
    events_on_latest_game = ctx.player.events.slice_from(last_ts)

    d = {
        'latest_events_count': len(events_on_latest_game),
    }
    return d


def _timestamp_match(df: TimeSeriesStream, ctx: Context) -> bool:
    if len(df) == 0:
        return False
    return df.last_timestamp() == ctx.daily_data_date


@feature(['player_' + c for c in PLAYER_BOX_FEATUERS])
def f050_player_box_scores_safe(ctx: Context) -> Dict:
    if not _timestamp_match(ctx.player.box_scores, ctx):
        return empty_feature(ctx.current_feature_name)

    d = {'player_' + c: ctx.player.box_scores.last_value(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_statusCode'])
def f051_player_rosters_2_safe(ctx: Context) -> Dict:
    if not _timestamp_match(ctx.player.rosters, ctx):
        return empty_feature(ctx.current_feature_name)

    d = {
        'player_statusCode': LABELS['statusCode'].get(ctx.player.rosters.last_value('statusCode'), 10)
    }
    return d


@feature(['player_transactions_typeCode'])
def f052_player_transactions_2_safe(ctx: Context) -> Dict:
    if not _timestamp_match(ctx.player.transactions, ctx):
        return empty_feature(ctx.current_feature_name)

    d = {
        'player_transactions_typeCode': LABELS['typeCode'].get(ctx.player.transactions.last_value('typeCode'), 16)
    }
    return d


@feature(['team_' + c for c in TEAM_BOX_FEATURES])
def f053_team_box_scores_safe(ctx: Context) -> Dict:
    team = ctx.team
    if team is None or not _timestamp_match(team.box_scores, ctx):
        d = {'team_' + k: None for k in TEAM_BOX_FEATURES}
    else:
        d = {'team_' + c: team.box_scores.last_value(c) for c in TEAM_BOX_FEATURES}

    return d


@feature(['team_game_wins', 'team_game_winPct', 'team_game_winner', 'team_game_score', 'team_game_isTie',
          'team_game_gameType', 'team_game_dayNight', 'team_game_diff_days'])
def f054_team_games_2_safe(ctx: Context) -> Dict:
    team = ctx.team

    if team is None or len(team.games) == 0 or not _timestamp_match(team.games, ctx):
        d = {
            'team_game_wins': None,
            'team_game_winPct': None,
            'team_game_winner': None,
            'team_game_score': None,
            'team_game_isTie': None,
            'team_game_gameType': None,
            'team_game_dayNight': None,
            'team_game_diff_days': None
        }
    else:
        d = {
            'team_game_wins': team.games['teamWins'][-1],
            'team_game_winPct': team.games['teamWinPct'][-1],
            'team_game_winner': team.games['teamWinner'][-1],
            'team_game_score': team.games['teamScore'][-1],
            'team_game_isTie': float(team.games['isTie'][-1]),
            'team_game_gameType': LABELS['gameType'].get(team.games['gameType'][-1], 6),
            'team_game_dayNight': int(team.games['dayNight'][-1] == 'day'),
            'team_game_diff_days': diff_days(ctx.daily_data_date, team.games)
        }
    return d


@feature([
    'latest_events_count', 'latest_events_nastyFactor_mean',
    'latest_events_startSpeed_max', 'latest_events_totalDistance_mean',
    'latest_events_totalDistance_max'
])
def f055_events_safe(ctx: Context) -> Dict:
    if not _timestamp_match(ctx.player.events, ctx):
        return empty_feature(ctx.current_feature_name)
    last_ts = ctx.player.events.index[-1]
    events_on_latest_game = ctx.player.events.slice_from(last_ts)

    d = {
        'latest_events_count': len(events_on_latest_game),
        'latest_events_nastyFactor_mean': events_on_latest_game.nanmean('nastyFactor'),
        'latest_events_startSpeed_max': events_on_latest_game.nanmax('startSpeed'),
        'latest_events_totalDistance_mean': events_on_latest_game.nanmean('totalDistance'),
        'latest_events_totalDistance_max': events_on_latest_game.nanmax('totalDistance')
    }
    return d


@feature([
    'latest_events_action_count', 'latest_events_pitch_count'
])
def f056_events_safe(ctx: Context) -> Dict:
    if not _timestamp_match(ctx.player.events, ctx):
        return empty_feature(ctx.current_feature_name)
    last_ts = ctx.player.events.index[-1]
    events_on_latest_game = ctx.player.events.slice_from(last_ts)

    mask = events_on_latest_game['type'] == 'action'

    d = {
        'latest_events_action_count': int(mask.sum()),
        'latest_events_pitch_count': int((~mask).sum())
    }
    return d


@feature(['player_' + c for c in PLAYER_BOX_FEATURES_2])
def f057_player_box_scores_safe(ctx: Context) -> Dict:
    if not _timestamp_match(ctx.player.box_scores, ctx):
        return empty_feature(ctx.current_feature_name)
    d = {'player_' + c: ctx.player.box_scores.last_value(c) for c in PLAYER_BOX_FEATURES_2}
    return d


@feature(['player_' + c + '_e' for c in PLAYER_BOX_FEATUERS])
def f058_player_box_scores_exact(ctx: Context) -> Dict:
    if len(ctx.player.box_scores) == 0 or (ctx.player.box_scores.index[-1] != ctx.daily_data_date):
        return empty_feature(ctx.current_feature_name)

    d = {'player_' + c + '_e': ctx.player.box_scores.last_value(c) for c in PLAYER_BOX_FEATUERS}
    return d
