from typing import Dict

import pandas as pd

from src.streamdf import TimeSeriesStream
from src.features import *


@feature([
    'events_action_count_ttl', 'events_pitch_count_ttl'
])
def f700_events(ctx: Context) -> Dict:
    if len(ctx.player.events) == 0:
        return empty_feature(ctx.current_feature_name)

    mask = ctx.player.events['type'] == 'action'

    d = {
        'events_action_count_ttl': int(mask.sum()),
        'events_pitch_count_ttl': int((~mask).sum())
    }
    return d


@feature([
    'events_action_count_per_box', 'events_pitch_count_per_box'
])
def f701_events_per_box(ctx: Context) -> Dict:
    if len(ctx.player.events) == 0:
        return empty_feature(ctx.current_feature_name)

    mask = ctx.player.events['type'] == 'action'
    n_box = len(ctx.player.box_scores)

    d = {
        'events_action_count_per_box': int(mask.sum()) / max(1, n_box),
        'events_pitch_count_per_box': int((~mask).sum()) / max(1, n_box)
    }
    return d


@feature([
    'events_injured', 'events_homerun', 'events_wildpitch'
])
def f702_events(ctx: Context) -> Dict:
    if len(ctx.player.events) == 0:
        return empty_feature(ctx.current_feature_name)
    last_ts = ctx.player.events.index[-1]
    events_on_latest_game = ctx.player.events.slice_from(last_ts)

    injured = (events_on_latest_game['event'] == 'Injury').sum()
    homerun = (events_on_latest_game['event'] == 'Home Run').sum()
    wild = (events_on_latest_game['event'] == 'Wild Pitch').sum()

    return {
        'events_injured': int(injured),
        'events_homerun': int(homerun),
        'events_wildpitch': int(wild)
    }


def _score_diff(e: TimeSeriesStream):
    home = e.last_value('homeScore')
    away = e.last_value('awayScore')
    if home is None or away is None:
        return None

    try:
        return home - away
    except:
        return None



@feature([
    'events_x0_min', 'events_x0_max'
])
def f707_events_x0(ctx: Context) -> Dict:
    if len(ctx.player.events) == 0:
        return empty_feature(ctx.current_feature_name)
    last_ts = ctx.player.events.index[-1]
    events_on_latest_game = ctx.player.events.slice_from(last_ts)

    # 最終イニング
    return {
        'events_x0_min': events_on_latest_game.nanmin('x0'),
        'events_x0_max': events_on_latest_game.nanmax('x0')
    }