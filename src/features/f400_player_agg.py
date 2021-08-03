# day-by-day features

from typing import Dict

import numpy as np

from src.features import *
from src.features.helper import nanmean
from src.features.helper import diff_days, day_diff
from src.player import PLAYER_BOX_FEATUERS


@feature(['player_' + c + '_sum_d30' for c in PLAYER_BOX_FEATUERS])
def f400_player_box_scores_sum_30days(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(30, 'D'))

    d = {'player_' + c + '_sum_d30': slice.sum(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_box_count_d30'])
def f401_player_box_scores_count_30days(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(30, 'D'))

    return {
        'player_box_count_d30': len(slice)
    }


@feature(['player_' + c + '_sum_d7' for c in PLAYER_BOX_FEATUERS])
def f402_player_box_scores_sum_7days(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(7, 'D'))

    d = {'player_' + c + '_sum_d7': slice.sum(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_' + c + '_sum_d90' for c in PLAYER_BOX_FEATUERS])
def f403_player_box_scores_sum_90days(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(90, 'D'))

    d = {'player_' + c + '_sum_d90': slice.sum(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_' + c + '_max_d30' for c in PLAYER_BOX_FEATUERS])
def f404_player_box_scores_max_30days(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(30, 'D'))

    d = {'player_' + c + '_max_d30': slice.max(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_box_count_d7'])
def f405_player_box_scores_count_7days(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(7, 'D'))

    return {
        'player_box_count_d7': len(slice)
    }


@feature(['player_box_count_d90'])
def f406_player_box_scores_count_90days(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(90, 'D'))

    return {
        'player_box_count_d90': len(slice)
    }


@feature(['player_box_count_d360'])
def f407_player_box_scores_count_360days(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(360, 'D'))

    return {
        'player_box_count_d360': len(slice)
    }


@feature(['player_' + c + '_mean_d30' for c in PLAYER_BOX_FEATUERS])
def f408_player_box_scores_mean_30days(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(30, 'D'))

    d = {'player_' + c + '_mean_d30': slice.mean(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_' + c + '_mean_d7' for c in PLAYER_BOX_FEATUERS])
def f409_player_box_scores_mean_7days(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(7, 'D'))

    d = {'player_' + c + '_mean_d30': slice.mean(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_' + c + '_sum_last3' for c in PLAYER_BOX_FEATUERS])
def f410_player_box_scores_sum_latest3(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.last_n(3)

    d = {'player_' + c + '_sum_last3': slice.sum(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_' + c + '_mean_last3' for c in PLAYER_BOX_FEATUERS])
def f411_player_box_scores_mean_latest3(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.last_n(3)

    d = {'player_' + c + '_mean_last3': slice.sum(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_' + c + '_sum_last10' for c in PLAYER_BOX_FEATUERS])
def f412_player_box_scores_sum_latest10(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.last_n(10)

    d = {'player_' + c + '_sum_last10': slice.sum(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_' + c + '_mean_last10' for c in PLAYER_BOX_FEATUERS])
def f413_player_box_scores_mean_latest10(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.last_n(10)

    d = {'player_' + c + '_mean_last10': slice.sum(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_' + c + '_sum_d360' for c in PLAYER_BOX_FEATUERS])
def f414_player_box_scores_sum_360days(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(360, 'D'))

    d = {'player_' + c + '_sum_d360': slice.sum(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_' + c + '_mean_d360' for c in PLAYER_BOX_FEATUERS])
def f415_player_box_scores_mean_360days(ctx: Context) -> Dict:
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(360, 'D'))

    d = {'player_' + c + '_mean_d360': slice.mean(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_' + c + '_diff_from_avg_360' for c in PLAYER_BOX_FEATUERS])
def f416_player_box_scores_diff_from_avg_360days(ctx: Context) -> Dict:
    if len(ctx.player.box_scores) == 0:
        return empty_feature(ctx.current_feature_name)
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(360, 'D'))

    d = {'player_' + c + '_diff_from_avg_360': slice.last_value(c) - slice.mean(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_' + c + '_diff_from_avg_30' for c in PLAYER_BOX_FEATUERS])
def f417_player_box_scores_diff_from_avg_30days(ctx: Context) -> Dict:
    if len(ctx.player.box_scores) == 0:
        return empty_feature(ctx.current_feature_name)
    slice = ctx.player.box_scores.slice_from(ctx.daily_data_date - np.timedelta64(30, 'D'))

    d = {'player_' + c + '_diff_from_avg_360': slice.last_value(c) - slice.mean(c) for c in PLAYER_BOX_FEATUERS}
    return d


@feature(['player_box_diff_days_2'])
def f418_player_box_scores_diff_last2games(ctx: Context) -> Dict:
    if len(ctx.player.box_scores) < 2:
        return empty_feature(ctx.current_feature_name)

    box = ctx.player.box_scores

    return {
        'player_box_diff_days_2': day_diff(box.index[-1], box.index[-2])
    }
