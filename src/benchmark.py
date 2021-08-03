
import copy
import os
import sys
import time
from contextlib import contextmanager


sys.path.append('../')

from src.feature import *
from src.store import *
from src.parser import *

@contextmanager
def timer(name):
    s = time.time()
    yield
    elapsed = time.time() - s
    print(f"[{name}] {elapsed:.3f}s")

TARGET_COLS = ['target1', 'target2', 'target3', 'target4']
DATA_DIR = '../input/mlb-player-digital-engagement-forecasting'
ARTIFACT_DIR = '../notebook/artifacts'


def bench():
    event_level_models = [
        lgb.Booster(model_file=os.path.join(ARTIFACT_DIR, f'meta_model_target{i}.bin')) for i in [1, 2]
    ]

    with timer('load'):
        store = Store.train(DATA_DIR, use_updated=True, event_model=event_level_models)


if __name__ == '__main__':
    bench()
