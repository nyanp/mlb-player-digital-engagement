import numbers
from datetime import datetime
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target


def check_cv(cv: Union[int, Iterable, BaseCrossValidator] = 5,
             y: Optional[Union[pd.Series, np.ndarray]] = None,
             stratified: bool = False,
             random_state: int = 0):
    if cv is None:
        cv = 5
    if isinstance(cv, numbers.Integral):
        if stratified and (y is not None) and (type_of_target(y) in ('binary', 'multiclass')):
            return StratifiedKFold(cv, shuffle=True, random_state=random_state)
        else:
            return KFold(cv, shuffle=True, random_state=random_state)

    return model_selection.check_cv(cv, y, stratified)


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

