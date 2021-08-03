from typing import Dict

import pandas as pd

from src.streamdf import TimeSeriesStream
from src.features import *
from src.features.helper import diff_days

@feature([
    'events_oof_target1_min',
    'events_oof_target1_max',
    'events_oof_target1_mean',
    'events_oof_target2_min',
    'events_oof_target2_max',
    'events_oof_target2_mean',
])
def f800_meta(ctx: Context) -> Dict:
    if len(ctx.player.events_meta) == 0:
        return empty_feature(ctx.current_feature_name)

    meta = ctx.player.events_meta

    return {
        k: meta.last_value(k) for k in [
            'events_oof_target1_min',
            'events_oof_target1_max',
            'events_oof_target1_mean',
            'events_oof_target2_min',
            'events_oof_target2_max',
            'events_oof_target2_mean',
        ]
    }


@feature([
    'events_oof_target1_g_max',
    'events_oof_target1_g_mean',
    'events_oof_target2_g_max',
    'events_oof_target2_g_mean'
])
def f801_meta_global(ctx: Context) -> Dict:
    if len(ctx.daily_stats) == 0:
        return empty_feature(ctx.current_feature_name)

    meta = ctx.daily_stats

    return {
        k: meta.last_value(k) for k in [
            'events_oof_target1_g_max',
            'events_oof_target1_g_mean',
            'events_oof_target2_g_max',
            'events_oof_target2_g_mean'
        ]
    }


@feature([
    'events_oof_target1_t_max',
    'events_oof_target1_t_mean',
    'events_oof_target2_t_max',
    'events_oof_target2_t_mean',
    'events_oof_target3_t_max',
    'events_oof_target3_t_mean',
    'events_oof_target4_t_max',
    'events_oof_target4_t_mean',
])
def f802_meta_team(ctx: Context) -> Dict:
    if ctx.team is None:
        return empty_feature(ctx.current_feature_name)

    meta = ctx.team.events_meta

    return {
        k: meta.last_value(k) for k in [
            'events_oof_target1_t_max',
            'events_oof_target1_t_mean',
            'events_oof_target2_t_max',
            'events_oof_target2_t_mean',
            'events_oof_target3_t_max',
            'events_oof_target3_t_mean',
            'events_oof_target4_t_max',
            'events_oof_target4_t_mean',
        ]
    }


@feature([
    'events_oof_target1_min',
    'events_oof_target1_max',
    'events_oof_target1_mean',
    'events_oof_target2_min',
    'events_oof_target2_max',
    'events_oof_target2_mean',
    'events_oof_target3_min',
    'events_oof_target3_max',
    'events_oof_target3_mean',
    'events_oof_target4_min',
    'events_oof_target4_max',
    'events_oof_target4_mean',
])
def f803_meta_asof_30d(ctx: Context) -> Dict:
    if len(ctx.player.events_meta) == 0:
        return empty_feature(ctx.current_feature_name)

    meta = ctx.player.events_meta

    if diff_days(ctx.daily_data_date, meta) > 30:
        return empty_feature(ctx.current_feature_name)

    return {
        k: meta.last_value(k) for k in [
            'events_oof_target1_min',
            'events_oof_target1_max',
            'events_oof_target1_mean',
            'events_oof_target2_min',
            'events_oof_target2_max',
            'events_oof_target2_mean',
            'events_oof_target3_min',
            'events_oof_target3_max',
            'events_oof_target3_mean',
            'events_oof_target4_min',
            'events_oof_target4_max',
            'events_oof_target4_mean',
        ]
    }


@feature([
    'events_oof_target1_t_max',
    'events_oof_target1_t_mean',
    'events_oof_target2_t_max',
    'events_oof_target2_t_mean',
    'events_oof_target3_t_max',
    'events_oof_target3_t_mean',
    'events_oof_target4_t_max',
    'events_oof_target4_t_mean',
])
def f804_meta_team_exact(ctx: Context) -> Dict:
    if ctx.team is None:
        return empty_feature(ctx.current_feature_name)

    meta = ctx.team.events_meta

    # exact match
    if len(meta) == 0 or meta.index[-1] != ctx.daily_data_date:
        return empty_feature(ctx.current_feature_name)

    return {
        k: meta.last_value(k) for k in [
            'events_oof_target1_t_max',
            'events_oof_target1_t_mean',
            'events_oof_target2_t_max',
            'events_oof_target2_t_mean',
            'events_oof_target3_t_max',
            'events_oof_target3_t_mean',
            'events_oof_target4_t_max',
            'events_oof_target4_t_mean',
        ]
    }


@feature([
    'events_oof_target1_g_max',
    'events_oof_target1_g_mean',
    'events_oof_target2_g_max',
    'events_oof_target2_g_mean',
    'events_oof_target3_g_max',
    'events_oof_target3_g_mean',
    'events_oof_target4_g_max',
    'events_oof_target4_g_mean',
])
def f805_meta_global_exact(ctx: Context) -> Dict:
    if len(ctx.daily_stats) == 0:
        return empty_feature(ctx.current_feature_name)

    meta = ctx.daily_stats

    # exact match
    if len(meta) == 0 or meta.index[-1] != ctx.daily_data_date:
        return empty_feature(ctx.current_feature_name)

    return {
        k: meta.last_value(k) for k in [
            'events_oof_target1_g_max',
            'events_oof_target1_g_mean',
            'events_oof_target2_g_max',
            'events_oof_target2_g_mean',
            'events_oof_target3_g_max',
            'events_oof_target3_g_mean',
            'events_oof_target4_g_max',
            'events_oof_target4_g_mean',
        ]
    }
