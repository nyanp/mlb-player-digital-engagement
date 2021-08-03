import numpy as np
from src.model_bank import *


def test_ensemble():
    gbdt = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])

    mixed = ensemble(0, gbdt, None, None)
    assert np.array_equal(gbdt, mixed)

    mixed = ensemble(-1, gbdt, None, None)
    assert np.array_equal(gbdt, mixed)


def test_ensemble2():
    gbdt = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0]
    ])

    cnn = np.array([
        [2, 4, 6, 8],
        [0, 0, 0, 0]
    ])

    mlp = np.array([
        [1, 1, 1, 1],
        [4, 3, 2, 1]
    ])

    # lag=0の時はGBDTだけ使う。他はNNも混ぜる。
    mixed = ensemble(0, gbdt, mlp, cnn)
    assert np.array_equal(gbdt, mixed)

    for lag in [3, 7, 14, 21, 28, 35, 45]:
        mixed = ensemble(lag, gbdt, mlp, cnn)
        assert not np.array_equal(gbdt, mixed)

    mixed = ensemble(28, gbdt, mlp, cnn)
    print(mixed)

    expected = np.zeros_like(gbdt)

    expected[:, 0] = (4*gbdt[:, 0] + cnn[:, 0]) / 5
    expected[:, 1] = (4*gbdt[:, 1] + cnn[:, 1] + mlp[:, 1]) / 6
    expected[:, 2] = (gbdt[:, 2] + cnn[:, 2] + mlp[:, 2]) / 3
    expected[:, 3] = (8*gbdt[:, 3] + cnn[:, 3] + mlp[:, 3]) / 10

    assert np.array_equal(expected, mixed)


def test_lag_manager():

    m = LagManager(np.datetime64('2021-01-01'), [7, 0, 3])

    assert m.get_current_lag(np.datetime64('2020-12-31')) == 0
    assert m.get_current_lag(np.datetime64('2021-01-01')) == 0
    assert m.get_current_lag(np.datetime64('2021-01-02')) == 0
    assert m.get_current_lag(np.datetime64('2021-01-03')) == 3
    assert m.get_current_lag(np.datetime64('2021-01-04')) == 3
    assert m.get_current_lag(np.datetime64('2021-01-05')) == 3
    assert m.get_current_lag(np.datetime64('2021-01-06')) == 7
    assert m.get_current_lag(np.datetime64('2021-01-07')) == 7
    assert m.get_current_lag(np.datetime64('2021-01-08')) == 7
    assert m.get_current_lag(np.datetime64('2022-01-01')) == 7


def test_lag_manager2():

    m = LagManager(np.datetime64('2021-07-31'), [7, 0, 3, 14, 21, 28, 35, 45])

    assert m.get_current_lag(np.datetime64('2021-07-31')) == 0
    assert m.get_current_lag(np.datetime64('2021-08-01')) == 0
    assert m.get_current_lag(np.datetime64('2021-08-02')) == 3
    assert m.get_current_lag(np.datetime64('2021-08-03')) == 3
    assert m.get_current_lag(np.datetime64('2021-08-04')) == 3
    assert m.get_current_lag(np.datetime64('2021-08-05')) == 7
    assert m.get_current_lag(np.datetime64('2021-08-06')) == 7
    assert m.get_current_lag(np.datetime64('2021-08-07')) == 7
    assert m.get_current_lag(np.datetime64('2021-08-08')) == 7
    assert m.get_current_lag(np.datetime64('2021-08-09')) == 14
    assert m.get_current_lag(np.datetime64('2021-08-10')) == 14
    assert m.get_current_lag(np.datetime64('2021-08-11')) == 14
    assert m.get_current_lag(np.datetime64('2021-08-12')) == 14
    assert m.get_current_lag(np.datetime64('2021-08-13')) == 14
    assert m.get_current_lag(np.datetime64('2021-08-14')) == 14
    assert m.get_current_lag(np.datetime64('2021-08-15')) == 14
    assert m.get_current_lag(np.datetime64('2021-08-16')) == 21
    assert m.get_current_lag(np.datetime64('2021-08-17')) == 21
    assert m.get_current_lag(np.datetime64('2021-08-18')) == 21
    assert m.get_current_lag(np.datetime64('2021-08-19')) == 21
    assert m.get_current_lag(np.datetime64('2021-08-20')) == 21
    assert m.get_current_lag(np.datetime64('2021-08-21')) == 21
    assert m.get_current_lag(np.datetime64('2021-08-22')) == 21
    assert m.get_current_lag(np.datetime64('2021-08-23')) == 28
    assert m.get_current_lag(np.datetime64('2021-08-24')) == 28
    assert m.get_current_lag(np.datetime64('2021-08-25')) == 28
    assert m.get_current_lag(np.datetime64('2021-08-26')) == 28
    assert m.get_current_lag(np.datetime64('2021-08-27')) == 28
    assert m.get_current_lag(np.datetime64('2021-08-28')) == 28
    assert m.get_current_lag(np.datetime64('2021-08-29')) == 28
    assert m.get_current_lag(np.datetime64('2021-08-30')) == 35
    assert m.get_current_lag(np.datetime64('2021-08-31')) == 35
    assert m.get_current_lag(np.datetime64('2021-09-01')) == 35
    assert m.get_current_lag(np.datetime64('2021-09-02')) == 35
    assert m.get_current_lag(np.datetime64('2021-09-03')) == 35
    assert m.get_current_lag(np.datetime64('2021-09-04')) == 35
    assert m.get_current_lag(np.datetime64('2021-09-05')) == 35
    assert m.get_current_lag(np.datetime64('2021-09-06')) == 45
