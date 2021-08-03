from typing import Optional

import numpy as np

from src.streamdf import TimeSeriesStream

try:
    import bottleneck

    nanmean = bottleneck.nanmean
    nanmin = bottleneck.nanmin
    nanmax = bottleneck.nanmax
    nanmedian = bottleneck.nanmedian
    median = bottleneck.median
    allnan = bottleneck.allnan
    nanstd = bottleneck.nanstd
except:
    nanmean = np.nanmean
    nanmin = np.nanmin
    nanmax = np.nanmax
    nanmedian = np.nanmedian
    median = np.median
    nanstd = np.nanstd


    def allnan(t):
        return np.all(np.isnan(t))


def diff_days(a: np.datetime64, b: TimeSeriesStream, n=-1) -> Optional[float]:
    if len(b) == 0:
        return None
    last_ts = b.last_timestamp(n)
    if last_ts is None:
        return None
    return (a - last_ts) / np.timedelta64(1, 'D')


def day_diff(a: np.datetime64, b: np.datetime64) -> Optional[float]:
    try:
        return (a - b) / np.timedelta64(1, 'D')
    except:
        return None
