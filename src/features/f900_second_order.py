# feature derived from other feature
import warnings
import traceback
from typing import List
import pandas as pd

from src.player import PLAYER_BOX_FEATUERS


def get_second_order_features() -> List[str]:
    return ['player_' + c + '_rank' for c in PLAYER_BOX_FEATUERS]


def f900_player_box_score_rank(src_df: pd.DataFrame, pred_base_df: pd.DataFrame) -> pd.DataFrame:
    dst_cols = get_second_order_features()
    try:
        assert len(src_df) == len(pred_base_df)

        src_cols = ['player_' + c for c in PLAYER_BOX_FEATUERS]

        ranks = src_df.groupby(pred_base_df['dailyDataDate'].values)[src_cols].rank()
        ranks.columns = dst_cols

        for c in dst_cols:
            src_df[c] = ranks[c].values

        return src_df
    except Exception:
        msg = f"Error in second order features! {traceback.format_exc()}"
        print(msg)
        warnings.warn(msg)
        for c in dst_cols:
            src_df[c] = None
        return src_df

