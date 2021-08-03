# lag features

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import skew

from src.constants import *
from src.features import *
from src.features.helper import nanmean, nanmax, nanmin, nanmedian, nanstd
from src.streamdf import TimeSeriesStream
from src.feature import *


def _range4(template) -> List[str]:
    return [template.format(i + 1) for i in range(4)]


def cnz(a):
    if len(a) == 0:
        return None
    return (a > 0).sum()


def make_lag_feature(ctx: Context,
                     engagement: TimeSeriesStream,
                     offset: int,
                     lag: int,
                     aggfunc = nanmean,
                     mask_days: np.ndarray = None,
                     mask_days_exclude: np.ndarray = None,
                     postfix = ''):
    #p = ctx.player
    #engagement = p.engagement

    if len(engagement):
        day_from = ctx.daily_data_date - np.timedelta64(offset + lag, 'D')
        day_to = ctx.daily_data_date - np.timedelta64(offset, 'D')
        mask = (day_from < engagement.primary_key) & (engagement.primary_key <= day_to)

        # 指定した日付に合致するものだけをとってくる
        if mask_days is not None:
            mask2 = np.in1d(engagement.primary_key, mask_days)
            mask = mask & mask2

        if mask_days_exclude is not None:
            mask2 = np.in1d(engagement.primary_key, mask_days_exclude)
            mask = mask & ~mask2

        if mask.sum() > 0:
            return {
                f"lag_target1_{lag}{postfix}": aggfunc(engagement["target1"][mask]),
                f"lag_target2_{lag}{postfix}": aggfunc(engagement["target2"][mask]),
                f"lag_target3_{lag}{postfix}": aggfunc(engagement["target3"][mask]),
                f"lag_target4_{lag}{postfix}": aggfunc(engagement["target4"][mask]),
            }
    return {
        f"lag_target1_{lag}{postfix}": None,
        f"lag_target2_{lag}{postfix}": None,
        f"lag_target3_{lag}{postfix}": None,
        f"lag_target4_{lag}{postfix}": None
    }


@feature(_range4('lag_target{}_28'), True)
def f100_lag_engagement_28(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 28)


@feature(_range4('lag_target{}_70'), True)
def f101_lag_engagement_70(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 70)


@feature(_range4('lag_target{}_140'), True)
def f102_lag_engagement_140(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 140)


@feature(_range4('lag_target{}_365'), True)
def f103_lag_engagement_365(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 365)


@feature(_range4('lag_target{}_720'), True)
def f104_lag_engagement_720(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 720)


@feature(_range4('lag_target{}_28_1y'))
def f105_lag_last_year_4w(ctx: Context) -> Dict:
    # 前年同月頃のlag
    return make_lag_feature(ctx, ctx.player.engagement, 351, 28, postfix='_1y')

## 条件付きLag

# gameがあった日のlag

@feature(_range4('lag_target{}_140_withgame'), True)
def f110_lag_engagement_withgame_140(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 140,
                            mask_days=ctx.player.box_scores.primary_key, postfix='_withgame')


@feature(_range4('lag_target{}_140_withoutgame'), True)
def f111_lag_engagement_withoutgame_140(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 140,
                            mask_days_exclude=ctx.player.box_scores.primary_key, postfix='_withoutgame')


@feature(_range4('lag_target{}_70_withoutgame'), True)
def f112_lag_engagement_withoutgame_70(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 70,
                            mask_days_exclude=ctx.player.box_scores.primary_key, postfix='_withoutgame')


## mean以外の集計


@feature(_range4('lag_target{}_70_min'), True)
def f120_lag_engagement_70_min(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 70, nanmin, postfix='_min')


@feature(_range4('lag_target{}_28_withoutgame_1y'))
def f121_lag_engagement_withoutgame_last_year(ctx: Context) -> Dict:
    recent_box = ctx.player.box_scores
    return make_lag_feature(ctx, ctx.player.engagement, 351, 28, mask_days_exclude=recent_box.primary_key, postfix='_withoutgame_1y')


@feature(_range4('lag_target{}_70_median'), True)
def f122_lag_engagement_70_median(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 70, nanmedian, postfix='_median')


@feature(_range4('lag_target{}_70_median'), True)
def f123_lag_engagement_70_max(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 70, nanmax, postfix='_max')


@feature(_range4('lag_target{}_720_onseason'), True)
def f130_lag_engagement_onseason_720(ctx: Context) -> Dict:
    mask = pd.to_datetime(ctx.player.engagement.primary_key).month.isin(ON_SEASON)
    valid_days = ctx.player.engagement.primary_key[mask]

    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements,
                            720, mask_days=valid_days, postfix='_onseason')


@feature(_range4('lag_target{}_720_onseason_withoutgame'), True)
def f131_lag_engagement_onseason_720_withoutgame(ctx: Context) -> Dict:
    mask = pd.to_datetime(ctx.player.engagement.primary_key).month.isin(ON_SEASON)
    valid_days = ctx.player.engagement.primary_key[mask]

    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements,
                            720, mask_days=valid_days,
                            mask_days_exclude=ctx.player.box_scores.primary_key,
                            postfix='_onseason_withoutgame')



@feature(_range4('lag_target{}_360_onseason'), True)
def f132_lag_engagement_onseason_360(ctx: Context) -> Dict:
    mask = pd.to_datetime(ctx.player.engagement.primary_key).month.isin(ON_SEASON)
    valid_days = ctx.player.engagement.primary_key[mask]

    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements,
                            360, mask_days=valid_days, postfix='_onseason')


@feature(_range4('lag_target{}_360_onseason_withoutgame'), True)
def f133_lag_engagement_onseason_360_withoutgame(ctx: Context) -> Dict:
    mask = pd.to_datetime(ctx.player.engagement.primary_key).month.isin(ON_SEASON)
    valid_days = ctx.player.engagement.primary_key[mask]

    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements,
                            360, mask_days=valid_days,
                            mask_days_exclude=ctx.player.box_scores.primary_key,
                            postfix='_onseason_withoutgame')


@feature(_range4('lag_target{}_720_onseason_withoutgame_2'), True)
def f134_lag_engagement_onseason_720_withoutgame_2(ctx: Context) -> Dict:
    mask = ctx.player.engagement['on_season']
    valid_days = ctx.player.engagement.primary_key[mask]

    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements,
                            720, mask_days=valid_days,
                            mask_days_exclude=ctx.player.box_scores.primary_key,
                            postfix='_onseason_withoutgame_2')


@feature(_range4('lag_target{}_35_min'), True)
def f135_lag_engagement_35_min(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 35, nanmin, postfix='_min')


@feature(_range4('lag_target{}_35_max'), True)
def f136_lag_engagement_35_max(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 35, nanmax, postfix='_max')


@feature(_range4('lag_target{}_35_std'), True)
def f137_lag_engagement_35_std(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 35, nanstd, postfix='_std')


@feature(_range4('lag_target{}_70_max'), True)
def f138_lag_engagement_70_max(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 70, nanmax, postfix='_max')


@feature(_range4('lag_target{}_70_std'), True)
def f139_lag_engagement_70_std(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 70, nanstd, postfix='_std')


@feature(_range4('lag_target{}_360_max'), True)
def f140_lag_engagement_360_max(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 360, nanmax, postfix='_max')


@feature(_range4('lag_target{}_360_std'), True)
def f141_lag_engagement_360_std(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 360, nanstd, postfix='_std')


@feature(_range4('lag_target{}_14'), True)
def f150_lag_engagement_14(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 14)


@feature(_range4('lag_target{}_7'), True)
def f151_lag_engagement_14(ctx: Context) -> Dict:
    return make_lag_feature(ctx, ctx.player.engagement, ctx.lag_requirements, 7)

