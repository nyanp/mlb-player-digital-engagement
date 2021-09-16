# Steps to reproduce the inference notebook

My final notebook can be seen [here](https://www.kaggle.com/nyanpn/3rd-place-solution-inference-only), 
but there are a few steps that need to be taken to recreate the private dataset I am using for the notebook.


Here are the steps to reproduce it. The specs of the GCP instance I used are as follows:

- Ubuntu 18.04 LTS
- vCPUx16, 128GB RAM
- 1 x Tesla V100

## 1. setup environment

```shell
> wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
> sh Anaconda3-2021.05-Linux-x86_64.sh
> tmux

> git clone https://github.com/nyanp/mlb-player-digital-engagement
> cd mlb-player-digital-engagement
> pip install -r requirements.txt
```

## 2. download competition data

Create an `input` directory, and place the competition data under it. 
If you are using [the official Kaggle API](https://github.com/Kaggle/kaggle-api), it should look like this:

```shell
> mkdir input
> cd input
> kaggle competitions download -c mlb-player-digital-engagement-forecasting
> unzip -d mlb-player-digital-engagement-forecasting mlb-player-digital-engagement-forecasting.zip
```

When the download and unzip is finished, the directory structure should look like this:

```shell
├── input/
│   └── mlb-player-digital-engagement-forecasting/
│       ├── train_updated.csv
│       └── ...
├── src/
│   ├── dummy_mlb/
│   ├── features/
│   └── ...
└── notebook/
    ├── 01_generate features.ipynb
    └── ...
```

## 3. run scripts
Run the notebooks in the following order.

- 01_generate features.ipynb
  - Note that this script takes over 100 hours to run on a 32core, 128GB RAM machine. 
    I split the loop into multiple notebooks and ran them in parallel on multiple machines.
- 02_event level model.ipynb
- 03_make nn dataset.ipynb
- 04_train lightgbm.ipynb
- 05_train nn.ipynb
  - Only this notebook is recommended to be run on a GPU-equipped environment.
- 06_ensemble.ipynb

## 4. upload artifacts to kaggle environment
After all the scripts have been run, you should have a trained model and some data files under the `artifacts/` folder. 
Upload this file as a Kaggle Dataset.

## 5. upload inference notebook
Upload `notebook/mlb-inference.ipynb` as a Kaggle notebook, and attach the dataset that you uploaded in step 4.
(During the competition, I was using GitHub Actions to automatically upload each commit)
