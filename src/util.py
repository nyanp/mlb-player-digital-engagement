import json
import os
import sys
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import BaseCrossValidator


_ALWAYS_CATCH = False


def in_colab() -> bool:
    return 'google.colab' in sys.modules


def in_kaggle() -> bool:
    return 'kaggle_web_client' in sys.modules


def set_always_catch(catch: bool):
    global _ALWAYS_CATCH
    _ALWAYS_CATCH = catch


@contextmanager
def catch_everything_in_kaggle(name: Optional[str] = None):
    try:
        yield
    except Exception:
        msg = f"WARNINGS: exception occur in {name or '(unknown)'}: {traceback.format_exc()}"
        warnings.warn(msg)
        print(msg)

        if in_kaggle() or _ALWAYS_CATCH:
            # ...catch and suppress if this is executed in kaggle
            pass
        else:
            # re-raise if this is executed outside of kaggle
            raise


def save_directory_as_kaggle_dataset(output_dir: str, title: str, id: str) -> None:
    cmd = f'kaggle datasets version -p {output_dir} -m "automatic upload" --dir-mode tar'

    metadata = {
        "title": title,
        "id": f"nyanpn/{id}",
        "licenses": [{"name": "CC0-1.0"}]
    }

    with open(os.path.join(output_dir, "dataset-metadata.json"), "w") as fp:
        json.dump(metadata, fp)

    os.system(cmd)


########################################################################################################################
# from nyaggle


def plot_importance(importance: pd.DataFrame, path: Optional[str] = None, top_n: int = 100,
                    figsize: Optional[Tuple[int, int]] = None,
                    title: Optional[str] = None):
    """
    Plot feature importance and write to image

    Args:
        importance:
            The dataframe which has "feature" and "importance" column
        path:
            The file path to be saved
        top_n:
            The number of features to be visualized
        figsize:
            The size of the figure
        title:
            The title of the plot
    Example:
        >>> import pandas as pd
        >>> import lightgbm as lgb
        >>> from nyaggle.util import plot_importance
        >>> from sklearn.datasets import make_classification

        >>> X, y = make_classification()
        >>> X = pd.DataFrame(X, columns=['col{}'.format(i) for i in range(X.shape[1])])
        >>> booster = lgb.train({'objective': 'binary'}, lgb.Dataset(X, y))
        >>> importance = pd.DataFrame({
        >>>     'feature': X.columns,
        >>>     'importance': booster.feature_importance('gain')
        >>> })
        >>> plot_importance(importance, 'importance.png')
    """
    importance = importance.groupby('feature')['importance'] \
        .mean() \
        .reset_index() \
        .sort_values(by='importance', ascending=False)

    if len(importance) > top_n:
        importance = importance.iloc[:top_n, :]

    if figsize is None:
        figsize = (10, 16)

    if title is None:
        title = 'Feature Importance'

    plt.figure(figsize=figsize)
    sns.barplot(x="importance", y="feature", data=importance)
    plt.title(title)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)


class TimeSeriesSplit(BaseCrossValidator):
    """ Time Series cross-validator

    Time Series cross-validator which provides train/test indices to split variable interval time series data.
    This class provides low-level API for time series validation strategy.
    This class is compatible with sklearn's ``BaseCrossValidator`` (base class of ``KFold``, ``GroupKFold`` etc).

    Args:
        source:
            The column name or series of timestamp.
        times:
            Splitting window, where times[i][0] and times[i][1] denotes train and test time interval in (i-1)th fold
            respectively. Each time interval should be pair of datetime or str, and the validator generates indices
            of rows where timestamp is in the half-open interval [start, end).
            For example, if ``times[i][0] = ('2018-01-01', '2018-01-03')``, indices for (i-1)th training data
            will be rows where timestamp value meets ``2018-01-01 <= t < 2018-01-03``.

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from nyaggle.validation import TimeSeriesSplit
        >>> df = pd.DataFrame()
        >>> df['time'] = pd.date_range(start='2018/1/1', periods=5)

        >>> folds = TimeSeriesSplit('time',
        >>>                         [(('2018-01-01', '2018-01-02'), ('2018-01-02', '2018-01-04')),
        >>>                          (('2018-01-02', '2018-01-03'), ('2018-01-04', '2018-01-06'))])

        >>> folds.get_n_splits()
        2

        >>> splits = folds.split(df)

        >>> train_index, test_index = next(splits)
        >>> train_index
        [0]
        >>> test_index
        [1, 2]

        >>> train_index, test_index = next(splits)
        >>> train_index
        [1]
        >>> test_index
        [3, 4]
    """
    datepair = Tuple[Union[datetime, str], Union[datetime, str]]

    def __init__(self, source: Union[pd.Series, str],
                 times: List[Tuple[datepair, datepair]] = None):
        self.source = source
        self.times = []
        if times:
            for t in times:
                self.add_fold(t[0], t[1])

    def _to_datetime(self, time: Union[str, datetime]):
        return time if isinstance(time, datetime) else pd.to_datetime(time)

    def _to_datetime_tuple(self, time: datepair):
        return self._to_datetime(time[0]), self._to_datetime(time[1])

    def add_fold(self, train_interval: datepair, test_interval: datepair):
        """
        Append 1 split to the validator.

        Args:
            train_interval:
                start and end time of training data.
            test_interval:
                start and end time of test data.
        """
        train_interval = self._to_datetime_tuple(train_interval)
        test_interval = self._to_datetime_tuple(test_interval)
        assert train_interval[1], "train_interval[1] should not be None"
        assert test_interval[0], "test_interval[0] should not be None"

        assert (not train_interval[0]) or (
                train_interval[0] <= train_interval[1]), "train_interval[0] < train_interval[1]"
        assert (not test_interval[1]) or (test_interval[0] <= test_interval[1]), "test_interval[0] < test_interval[1]"

        self.times.append((train_interval, test_interval))

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.times)

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Args:
            X:
                Training data.
            y:
                Ignored.
            groups:
                Ignored.

        Yields:
            The training set and the testing set indices for that split.
        """
        ts = X[self.source] if isinstance(self.source, str) else self.source

        for train_interval, test_interval in self.times:
            train_mask = ts < train_interval[1]
            if train_interval[0]:
                train_mask = (train_interval[0] <= ts) & train_mask

            test_mask = test_interval[0] <= ts
            if test_interval[1]:
                test_mask = test_mask & (ts < test_interval[1])

            yield np.where(train_mask)[0], np.where(test_mask)[0]
