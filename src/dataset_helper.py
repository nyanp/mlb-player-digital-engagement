import json
import os

import pandas as pd
from tqdm import tqdm


def load_train_data(data_dir: str, use_updated: bool):
    stem = 'train_updated' if use_updated else 'train'
    path = os.path.join(data_dir, f'{stem}.f')

    if not os.path.exists(path):
        print(f'dataset {path} does not exist. try to create...')

        df = pd.read_csv(os.path.join(data_dir, f'{stem}.csv'))
        df.to_feather(path)

    return pd.read_feather(path)


def load_subdata(data_dir: str, name: str, use_updated: bool):
    postfix = '_updated' if use_updated else ''
    path = os.path.join(data_dir, f'train_{name}{postfix}.f')

    if not os.path.exists(path):
        print(f'dataset {path} does not exist. try to create...')

        train = load_train_data(data_dir, use_updated)

        eng = []
        for i, row in tqdm(train.iterrows()):
            try:
                loaded = json.loads(row[name])
                for l in loaded:
                    l['dailyDataDate'] = row.date
                eng.extend(loaded)
            except:
                pass

        events = pd.DataFrame(eng)
        assert len(events)
        events.to_feather(path)
        del train

    return pd.read_feather(path)
