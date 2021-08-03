from typing import Dict

import pandas as pd

from src.features import *


@feature(['dayofweek'])
def f600_dayofweek(ctx: Context) -> Dict:
    return {'dayofweek': pd.to_datetime(ctx.daily_data_date).dayofweek}

