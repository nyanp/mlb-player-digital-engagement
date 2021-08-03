# day-by-day features

from typing import Dict

import pandas as pd
import numpy as np

from src.features import *
from src.features.helper import nanmean
from src.features.helper import diff_days, day_diff


def _to_int(v):
    if pd.isnull(v):
        return None
    try:
        return int(v)
    except:
        return None


@feature(['daily_number_of_games', 'days_from_last_game'])
def f300_number_of_games(ctx: Context) -> Dict:
    return {
        "daily_number_of_games": _to_int(ctx.daily_stats.last_value("daily_number_of_games")),
        "days_from_last_game": diff_days(ctx.daily_data_date, ctx.daily_stats)
    }


@feature(['daily_number_of_awards'])
def f301_number_of_awards(ctx: Context) -> Dict:
    return {
        "daily_number_of_awards": _to_int(ctx.daily_stats.last_value("daily_number_of_awards"))
    }


@feature(['daily_number_of_transactions'])
def f302_number_of_transactions(ctx: Context) -> Dict:
    return {
        "daily_number_of_transactions": _to_int(ctx.daily_stats.last_value("daily_number_of_transactions"))
    }


@feature(['daily_number_of_events'])
def f303_number_of_events(ctx: Context) -> Dict:
    return {
        "daily_number_of_events": _to_int(ctx.daily_stats.last_value("daily_number_of_events"))
    }


@feature(['days_from_last_game_2'])
def f304_number_of_games(ctx: Context) -> Dict:

    n_games_mask = ctx.daily_stats["daily_number_of_games"] > 0
    indices = ctx.daily_stats.index[n_games_mask]

    if len(indices) == 0:
        return empty_feature(ctx.current_feature_name)

    return {
        "days_from_last_game_2": day_diff(ctx.daily_data_date, indices[-1])
    }

