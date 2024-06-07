import fds
import pandas as pd
import numpy as np
from datasets.function import standardize_and_split
from torch.utils.data import DataLoader, Dataset , TensorDataset
import torch 
from model import LSTM
from utilities import *
import os
import re
import shutil
# test_begin_date = '2015-01-03'
test_begin_date = '2019-06-01'
test_end_date = '2020-06-30'
threshold_of_grow_and_decline = 0.5

status = fds.list_status(test_begin_date, test_end_date, columns=None)
bars = fds.bar(test_begin_date, test_end_date, freq='1d')
flager = fds.bar(test_begin_date, test_end_date, freq='5m')
flager['close_ge_high'] = (flager['close'] >= flager['high']).astype(int)
flager['close_le_low'] = (flager['close'] <= flager['low']).astype(int)
flags_high_ratio = flager.groupby(['date', 'symbol'])['close_ge_high'].transform('mean')
flags_low_ratio = flager.groupby(['date', 'symbol'])['close_le_low'].transform('mean')
flager['flag'] = ((flags_high_ratio > threshold_of_grow_and_decline) | (flags_low_ratio > threshold_of_grow_and_decline)).astype(int)
daily_flags = (flager.groupby(['date', 'symbol'])['flag']
            .first()
            .reset_index())
pools = {}
merged = pd.merge(bars, status, on=['date', 'symbol'], how='left')
merged = pd.merge(merged, daily_flags, on=['date', 'symbol'], how='left')

filtered = merged[(merged['PT'] != 1) & (merged['ST'] != 1) & (merged['turnover'] != 0) & (merged['flag'] != 1)]
pools = {date: symbols.reset_index(drop=True) for date, symbols in filtered.groupby('date')['symbol']}
pools_df = pd.concat(pools.values(), keys=pools.keys()).reset_index().drop('level_1', axis=1)
pools_df.columns = ['date', 'symbol']

output_dim = 1
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

model_dir = 'models/'
for model_name in os.listdir(model_dir):
    if model_name.endswith('val.pth'):
        match = re.search(r'input_size_(\d+)_prediction_size_(\d+)_window_(\d+)_hidden_dim_(\d+)_num_layers_(\d+)_epoch_(\d+)', model_name)
        if (match and int(match.group(1))==18 and int(match.group(6))==12):
            print ("backtest data generate begin: {}".format(model_name))
            input_size = int(match.group(1))
            prediction_size = int(match.group(2))
            window = int(match.group(3))
            hidden_dim = int(match.group(4))
            num_layers = int(match.group(5))
            epoch = int(match.group(6))
            factor_name = match.group(0)
            backtest_data_generate(
                    bars=bars, pools_df=pools_df,
                    input_size=input_size, output_dim=output_dim,
                    prediction_size=prediction_size,
                    window=window,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    model_name= model_dir + model_name,
                    factor_name=factor_name,
                    device=device)

            factor_dir = os.path.join(os.path.expanduser('~'), 'backtest/factors', factor_name)
            if not os.path.exists(factor_dir):
                os.makedirs(factor_dir)

            data_file = os.path.join(os.getcwd(), 'factor', f'{factor_name}.parq')
            destination_file = os.path.join(factor_dir, 'data.parq')
            if os.path.isfile(data_file):
                shutil.copyfile(data_file, destination_file)
                print ("backtest data generate end: {}".format(model_name))
            else:
                print(f"data file {data_file} does not exist.")