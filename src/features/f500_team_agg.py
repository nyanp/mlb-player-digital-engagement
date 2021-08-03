from typing import Dict

import numpy as np

from src.features import *
from src.features.helper import nanmean
from src.features.helper import diff_days
from src.team import TEAM_BOX_FEATURES


@feature(['team_' + c + '_sum_d30' for c in TEAM_BOX_FEATURES])
def f500_team_box_scores_sum_30days(ctx: Context) -> Dict:
    if ctx.team is None:
        return empty_feature(ctx.current_feature_name)

    slice = ctx.team.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(30, 'D'))

    d = {'team_' + c + '_sum_d30': slice.sum(c) for c in TEAM_BOX_FEATURES}
    return d


@feature(['team_' + c + '_sum_d7' for c in TEAM_BOX_FEATURES])
def f501_team_box_scores_sum_7days(ctx: Context) -> Dict:
    if ctx.team is None:
        return empty_feature(ctx.current_feature_name)

    slice = ctx.team.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(7, 'D'))

    d = {'team_' + c + '_sum_d7': slice.sum(c) for c in TEAM_BOX_FEATURES}
    return d


@feature(['team_box_count_d360'])
def f502_team_box_scores_count_360days(ctx: Context) -> Dict:
    if ctx.team is None:
        return empty_feature(ctx.current_feature_name)

    slice = ctx.team.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(360, 'D'))

    return {
        'team_box_count_d360': len(slice)
    }


@feature(['team_box_count_d30'])
def f503_team_box_scores_count_30days(ctx: Context) -> Dict:
    if ctx.team is None:
        return empty_feature(ctx.current_feature_name)

    slice = ctx.team.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(30, 'D'))

    return {
        'team_box_count_d30': len(slice)
    }


@feature(['team_box_count_d7'])
def f504_team_box_scores_count_7days(ctx: Context) -> Dict:
    if ctx.team is None:
        return empty_feature(ctx.current_feature_name)

    slice = ctx.team.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(7, 'D'))

    return {
        'team_box_count_d7': len(slice)
    }