# features related to player/team twitter stats

from typing import Dict

import numpy as np

from src.features import *
from src.streamdf import TimeSeriesStream


def followers_features(followers: TimeSeriesStream, reference_date: np.datetime64, days: int, feature_prefix: str) -> Dict:
    sliced = followers.slice_from(reference_date - np.timedelta64(days, 'D'))
    return {
        f'{feature_prefix}_followersChangedIn{days}days': sliced.last_minus_first_value('numberOfFollowers'),
        f'{feature_prefix}_activitiesIn{days}days': len(sliced)
    }


@feature(['player_followersDiff', 'player_followersDiffRatio'])
def f204_player_twitter_followers_diff(ctx: Context) -> Dict:
    d = {
    }
    followers = ctx.player.followers

    if len(ctx.player.followers) > 1:
        d['player_followersDiff'] = followers['numberOfFollowers'][-1] - followers['numberOfFollowers'][-2]
        d['player_followersDiffRatio'] = followers['numberOfFollowers'][-1] / max(1, followers['numberOfFollowers'][-2])
    else:
        d['player_followersDiff'] = None
        d['player_followersDiffRatio'] = None

    return d

