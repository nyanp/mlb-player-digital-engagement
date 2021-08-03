import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
import numpy as np


class StreamDf:
    # pandas-likeなアクセスを提供するnumpy arrayのwrapper
    def __init__(self,
                 values: Dict[str, np.ndarray],
                 length: int = None,
                 primary_key_name: str = None):
        self._values = dict(values)
        self._capacity = len(list(values.values())[0])
        self._len = length if length is not None else self._capacity
        self.columns = list(self._values.keys())
        self.primary_key_name = primary_key_name

    def copy(self):
        return StreamDf(
            copy.deepcopy(self._values),
            self._len
        )

    @classmethod
    def empty(cls, column_schema: Dict[str, Type], primary_key_name: str = None, primary_key_dtype: Type = None):
        values = {
            k: np.empty(0, dtype=v) for k, v in column_schema.items()
        }
        if primary_key_name is not None:
            values[primary_key_name] = np.empty(0, dtype=primary_key_dtype)
        return cls(values, primary_key_name=primary_key_name)

    def __getitem__(self, item: Union[int, str, Tuple]):
        try:
            if isinstance(item, str):
                return self._values[item][:self._len]  # type: np.ndarray
            elif isinstance(item, np.ndarray):
                # boolean indexer
                return StreamDf({
                    k: v[item] for k, v in self._values.items()
                }, None)
            else:
                raise NotImplementedError()
        except KeyError:
            print(f'Key not found: {item} (columns: {self.columns})')
            raise

    def __len__(self):
        return self._len

    def masked(self,
               mask: np.ndarray,
               always_df: bool = False) -> Union['StreamDf', Dict]:
        if mask.sum() == 1 and not always_df:
            idx = np.where(mask)[0][0]
            values = {k: self._values[k][idx] for k in self.columns}
            return values
        else:
            values = {k: self[k][mask] for k in self.columns}
            return StreamDf(values)

    def sliced(self, len: int):
        return StreamDf(self._values, length=min(len, self._len), primary_key_name=self.primary_key_name)

    def _grow(self, min_capacity):
        capacity = max(int(1.5 * self._capacity), min_capacity)
        new_data_len = capacity - self._capacity
        assert new_data_len > 0

        for k in self._values:
            self._values[k] = np.concatenate([
                self._values[k],
                np.empty(new_data_len, dtype=self._values[k].dtype)
            ])
        self._capacity += new_data_len

    def _extend_series(self, df: pd.Series, primary_key_value: Any = None):
        if self._len + 1 > self._capacity:
            self._grow(self._len + 1)

        for c in self.columns:
            if self.primary_key_name is not None and c == self.primary_key_name:
                self._values[self.primary_key_name][self._len] = primary_key_value
                continue

            assert c in df.index
            self._values[c][self._len] = df[c]

        self._len += 1

    def _extend_dict(self, df: Dict[str, Any], primary_key_value: Any = None):
        if self._len + 1 > self._capacity:
            self._grow(self._len + 1)

        keys = df.keys()
        for c in self.columns:
            if self.primary_key_name is not None and c == self.primary_key_name:
                if primary_key_value is None and c in df:
                    primary_key_value = df[c]
                self._values[self.primary_key_name][self._len] = primary_key_value
                continue

            if c not in keys:
                warnings.warn(f"{c} not found in keys {keys}")
                self._values[c][self._len] = None
                continue

            try:
                value = df[c]
                self._values[c][self._len] = value
            except (TypeError, ValueError):
                print(f'expected type: {self._values[c].dtype}, actual value: {df[c]} in column {c}')
                self._values[c][self._len] = None

        self._len += 1

    def _extend_df(self, df: 'StreamDf', primary_key_value: Any = None):
        new_data_len = len(df)
        if new_data_len == 0:
            return

        if self._len + new_data_len > self._capacity:
            self._grow(self._len + new_data_len)

        for c in self.columns:
            assert c in df.columns
            self._values[c][self._len:self._len + new_data_len] = df[c]

        if self.primary_key_name is not None:
            self._values[self.primary_key_name][self._len:self._len + new_data_len] = primary_key_value

        self._len += new_data_len

    def extend(self,
               df: Union['StreamDf', pd.Series, Dict], primary_key_value: Any = None):
        if isinstance(df, dict):
            self._extend_dict(df, primary_key_value)
        elif isinstance(df, pd.Series):
            self._extend_series(df, primary_key_value)
        else:
            self._extend_df(df, primary_key_value)

    def extend_raw(self, d: Dict[str, np.ndarray]):
        new_data_len = len(d[self.columns[0]])

        if self._len + new_data_len > self._capacity:
            self._grow(self._len + new_data_len)

        for c in self.columns:
            self._values[c][self._len:self._len + new_data_len] = d[c]

        self._len += new_data_len

    @property
    def shape(self):
        return self._len, len(self._values.keys())

    @property
    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame({c: self[c] for c in self.columns})
        return df

    @property
    def primary_key(self) -> np.ndarray:
        return self[self.primary_key_name]

    @property
    def index(self) -> np.ndarray:
        return self.primary_key

    @classmethod
    def from_pandas(cls,
                    df: pd.DataFrame,
                    columns=None):
        columns = columns if columns is not None else df.columns
        return cls({
            c: df[c].values for c in columns
        })

    def first_value(self, column):
        if len(self) == 0:
            return None
        return self[column][0]

    def last_value(self, column):
        if len(self) == 0:
            return None
        return self[column][-1]

    def last_minus_first_value(self, column):
        if len(self) == 0:
            return None
        try:
            return self[column][-1] - self[column][0]
        except Exception:
            return None

    def sum(self, column):
        if len(self) == 0:
            return None
        try:
            return self[column].sum()
        except Exception:
            return None

    def mean(self, column):
        if len(self) == 0:
            return None
        try:
            return self[column].mean()
        except:
            return None

    def max(self, column):
        if len(self) == 0:
            return None
        try:
            return self[column].max()
        except Exception:
            return None

    def min(self, column):
        if len(self) == 0:
            return None
        try:
            return self[column].min()
        except:
            return None

    def _reduce(self, column, f):
        if len(self) == 0:
            return None
        try:
            if np.all(self[column] != self[column]):
                return None
            return f(self[column])
        except Exception:
            return None

    def nanmean(self, column):
        return self._reduce(column, np.nanmean)

    def nanmax(self, column):
        return self._reduce(column, np.nanmax)

    def nanmin(self, column):
        return self._reduce(column, np.nanmin)

    def nansum(self, column):
        return self._reduce(column, np.nansum)

    def argmax(self, column):
        if len(self) == 0:
            return None
        try:
            return np.argmax(self[column])
        except Exception:
            return None


class TimeSeriesStream(StreamDf):
    def __init__(self,
                 values: Dict[str, np.ndarray],
                 length: int = None,
                 primary_key_name: str = None):
        super().__init__(values, length, primary_key_name)

    @classmethod
    def empty(cls, column_schema: Dict[str, Type], primary_key_name: str = None):
        values = {
            k: np.empty(0, dtype=v) for k, v in column_schema.items()
        }
        values[primary_key_name] = np.empty(0, dtype='datetime64[D]')
        return cls(values, primary_key_name=primary_key_name)

    def recent_n_days(self, n: int, base: np.datetime64):
        th = base - np.timedelta64(n, 'D')
        return self.masked(self.primary_key >= th)

    def slice_until(self, until: np.datetime64) -> 'TimeSeriesStream':
        # ts <= untilまでのデータでスライスする.
        index = np.searchsorted(self.primary_key, until, side='right')
        return TimeSeriesStream(self._values, index, self.primary_key_name)

    def slice_from(self, from_: np.datetime64) -> 'TimeSeriesStream':
        l = np.searchsorted(self.primary_key, from_, side='left')
        if l >= self._len:
            return TimeSeriesStream(self._values, 0, self.primary_key_name)
        values = {
            k: v[l:self._len] for k, v in self._values.items()
        }
        return TimeSeriesStream(values, primary_key_name=self.primary_key_name)

    def last_n(self, n: int) -> 'TimeSeriesStream':
        n = min(n, len(self))

        if n == 0:
            return TimeSeriesStream(self._values, 0, self.primary_key_name)

        values = {
            k: v[self._len-n:self._len] for k, v in self._values.items()
        }
        return TimeSeriesStream(values, primary_key_name=self.primary_key_name)

    def slice_between(self, from_: np.datetime64, until: np.datetime64) -> 'TimeSeriesStream':
        # from_ ~ untilまで（当日を含む）のデータにスライスする
        r = np.searchsorted(self.primary_key, until, side='right')
        r = min(r, self._len)
        l = np.searchsorted(self.primary_key, from_, side='left')

        if l >= r:
            return TimeSeriesStream(self._values, 0, self.primary_key_name)

        assert l < r

        values = {
            k: v[l:r] for k, v in self._values.items()
        }
        return TimeSeriesStream(values, primary_key_name=self.primary_key_name)

    def last_timestamp(self, n=-1):
        if len(self) < -n:
            return None
        return self.index[n]
