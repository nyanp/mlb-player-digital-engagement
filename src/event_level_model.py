import pandas as pd


def prep_events(events: pd.DataFrame, sort_by_date: bool = True):
    events_p = events.copy()
    events_p['asPitcher'] = 1
    events_p['playerId'] = events_p['pitcherId']
    events_p['teamId'] = events_p['pitcherTeamId']

    events_h = events.copy()
    events_h['asPitcher'] = 0
    events_h['playerId'] = events_p['hitterId']
    events_h['teamId'] = events_p['hitterTeamId']

    if sort_by_date:
        events_stacked = pd.concat([events_p, events_h]).sort_values(
            by=['dailyDataDate', 'gamePk', 'inning']).reset_index(
            drop=True)
        events_stacked['dailyDataDate'] = pd.to_datetime(events_stacked['dailyDataDate'], format='%Y%m%d')
    else:
        events_stacked = pd.concat([events_p, events_h]).reset_index(drop=True)

    drop_cols = [
        'gameDate', 'season', 'playId', 'pitcherTeam', 'hitterTeam', 'pitcherName', 'pitcherHand', 'hitterName',
        'batSide', 'atBatDesc', 'gameTimeUTC',
        'description', 'halfInning', 'hitterTeamId', 'pitcherTeamId', 'gamePk', 'pitcherId', 'hitterId'
    ]

    events_stacked.drop(drop_cols, axis=1, inplace=True)

    events_stacked['scoreDiff'] = events_stacked['homeScore'] - events_stacked['awayScore']

    cats = {
        'menOnBase': {
            'Empty': 1,
            'Men_On': 2,
            'RISP': 3,
            'Loaded': 4
        },
        'gameType': {
            'R': 1,
            'D': 2,
            'L': 3,
            'F': 4,
            'W': 5
        },
        'atBatEvent': {
            'Strikeout': 1,
            'Groundout': 2,
            'Single': 3,
            'Walk': 4,
            'Flyout': 5,
            'Lineout': 6,
            'Pop Out': 7,
            'Double': 8,
            'Home Run': 9,
            'Forceout': 10,
            'Grounded Into DP': 11,
            'Hit By Pitch': 12,
            'Field Error': 13,
            'Sac Fly': 14,
            'Intent Walk': 15,
            'Triple': 16,
            'Double Play': 17,
            'Sac Bunt': 18,
            'Fielders Choice Out': 19,
            'Fielders Choice': 20,
            'Strikeout Double Play': 21,
            'Caught Stealing 2B': 22,
            'Bunt Groundout': 23,
            'Catcher Interference': 24,
            'Bunt Pop Out': 25,
            'Batter Interference': 26,
            'Runner Out': 27,
            'Pickoff Caught Stealing 2B': 28,
            'Fan Interference': 29,
            'Pickoff 1B': 30,
            'Caught Stealing 3B': 31,
            'Caught Stealing Home': 32,
            'Pickoff 2B': 33,
            'Sac Fly Double Play': 34,
            'Bunt Lineout': 35,
            'Wild Pitch': 36,
            'Pickoff Caught Stealing Home': 37,
            'Triple Play': 38,
            'Pickoff Caught Stealing 3B': 39,
            'Pickoff 3B': 40,
            'Game Advisory': 41,
            'Stolen Base 2B': 42,
            'Sac Bunt Double Play': 43,
            'Runner Double Play': 44,
            'Passed Ball': 45,
            'Pickoff Error 1B': 46,
            'Balk': 47
        },
        'event': {
            'Strikeout': 1,
            'Groundout': 2,
            'Single': 3,
            'Game Advisory': 4,
            'Flyout': 5,
            'Pitching Substitution': 6,
            'Walk': 7,
            'Lineout': 8,
            'Pop Out': 9,
            'Double': 10,
            'Home Run': 11,
            'Offensive Substitution': 12,
            'Defensive Switch': 13,
            'Forceout': 14,
            'Grounded Into DP': 15,
            'Defensive Sub': 16,
            'Hit By Pitch': 17,
            'Stolen Base 2B': 18,
            'Wild Pitch': 19,
            'Field Error': 20,
            'Sac Fly': 21,
            'Intent Walk': 22,
            'Triple': 23,
            'Sac Bunt': 24,
            'Caught Stealing 2B': 25,
            'Double Play': 26,
            'Passed Ball': 27,
            'Injury': 28,
            'Fielders Choice Out': 29,
            'Fielders Choice': 30,
            'Stolen Base 3B': 31,
            'Defensive Indiff': 32,
            'Bunt Groundout': 33,
            'Ejection': 34,
            'Balk': 35,
            'Strikeout Double Play': 36,
            'Runner Placed On Base': 37,
            'Pickoff Error 1B': 38,
            'Bunt Pop Out': 39,
            'Pitch Challenge': 40,
            'Runner Out': 41,
            'Pickoff 1B': 42,
            'Pickoff Caught Stealing 2B': 43,
            'Caught Stealing 3B': 44,
            'Catcher Interference': 45,
            'Error': 46,
            'Umpire Substitution': 47,
            'Batter Interference': 48,
            'Pickoff Error 2B': 49,
            'Pitcher Switch': 50,
            'Fan Interference': 51,
            'Pickoff 2B': 52,
            'Caught Stealing Home': 53,
            'Stolen Base Home': 54,
            'Sac Fly Double Play': 55,
            'Bunt Lineout': 56,
            'Pickoff Caught Stealing 3B': 57,
            'Pickoff 3B': 58,
            'Pickoff Caught Stealing Home': 59,
            'Other Advance': 60,
            'Pickoff Error 3B': 61,
            'Triple Play': 62,
            'Sac Bunt Double Play': 63},
        'pitchType': {'FF': 1,
                      'SL': 2,
                      'CH': 3,
                      'SI': 4,
                      'CU': 5,
                      'FT': 6,
                      'FC': 7,
                      'KC': 8,
                      'FS': 9,
                      'KN': 10,
                      'EP': 11,
                      'CS': 12,
                      'FO': 13,
                      'PO': 14,
                      'SC': 15,
                      'FA': 16,
                      'AB': 17},
        'call': {'B': 1,
                 'F': 2,
                 'C': 3,
                 'X': 4,
                 'S': 5,
                 'D': 6,
                 '*B': 7,
                 'E': 8,
                 'T': 9,
                 'W': 10,
                 'V': 11,
                 'H': 12,
                 'L': 13,
                 'M': 14,
                 'P': 15,
                 'O': 16,
                 'Q': 17,
                 'R': 18},
        'type': {
            'pitch': 1,
            'action': 2
        }
    }

    for k in cats:
        events_stacked[k] = events_stacked[k].map(cats[k])

    return events_stacked
