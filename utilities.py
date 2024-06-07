import numpy as np
from factors import *
import torch
import fds
import pandas as pd
from datasets.function import standardize_and_split
from torch.utils.data import DataLoader, Dataset , TensorDataset
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from scipy import stats
from model import *
from utilities import *
import os
import logging
import random
random.seed(10086)
def add_factors(df):
    df = Alpha088_to_df(df)
    df = Alpha084_to_df(df)
    df = Alpha080_to_df(df)
    df = Alpha071_to_df(df)
    df = Alpha070_to_df(df)
    df = Alpha069_to_df(df)
    ##############################18
    df = Alpha060_to_df(df)
    df = Alpha059_to_df(df)
    df = Alpha046_to_df(df)
    df = Alpha040_to_df(df)
    df = Alpha031_to_df(df)
    df = Alpha029_to_df(df)
    df = Alpha027_to_df(df)
    # df = Alpha026_to_df(df) 
    return df

def backtest_data_generate(bars, pools_df,
            input_size , output_dim = 1,
            prediction_size = 1,
            window = 20,
            hidden_dim = 64,
            num_layers = 2,
            model_name="models/prediction_size_1_window_20_hidden_dim_256_num_layers_2_train.pth",
            factor_name = 'prediction_size_1_window_20_hidden_dim_64_num_layers_2',
            device = 'cpu',
            adds = False):
    
    # 添加预测日收益率 标准化 数据集拆分
    df = bars
    df['ret'] = df['close']/df['pre_close'] - 1
    if adds:
        df = add_factors(df)

    test_X , test_y , dates , symbols = standardize_and_split(df, prediction_size, pools_df, window)

    flattened_date = np.concatenate(dates)
    flattened_symbol = np.concatenate(symbols)
    df = pd.DataFrame({
        'date': flattened_date,
        'symbol': flattened_symbol
    })
    test_dataset = TensorDataset(test_X, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=True)

    model = LSTM(input_size, hidden_dim, num_layers, output_dim).to(device)

    checkpoint = torch.load(model_name, map_location = device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with torch.no_grad():
        outputs = []
        for batch_idx, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            outputs.append(model(data))

    df[factor_name] = torch.cat(outputs).cpu().numpy()
    df.to_parquet('factor/{}.parq'.format(factor_name))

def backtest_data_generate_dict(df_keys,dataset,
            input_size , output_dim = 1,
            window = 20,
            hidden_dim = 64,
            num_layers = 2,
            model_name="models/prediction_size_1_window_20_hidden_dim_256_num_layers_2_train.pth",
            factor_name = 'prediction_size_1_window_20_hidden_dim_64_num_layers_2',
            device = 'cpu',
            dropout_rate = 0.05,
            c = None,
            num_heads = 2):
    if c == None:
        model = gru(input_size, hidden_dim, num_layers, output_dim).to(device)
    else:
        model = CrossGRU(input_size, hidden_dim, num_layers, output_dim, dropout_rate, c, num_heads, device).to(device)
    checkpoint = torch.load(model_name, map_location = device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.no_grad():
        outputs = []

        keys = list(dataset.keys())
        for key in keys:
            data, target = dataset[key].tensors[0] , dataset[key].tensors[1]
            data, target = data.to(device), target.to(device)
            target = target.view(-1)
            outputs.append(model(data))
    df_keys[factor_name] = torch.cat(outputs).cpu().numpy()
    df_keys[factor_name] = df_keys.groupby('symbol')[factor_name].shift(1)
    df_keys = df_keys.reset_index(drop = True)
    df_keys = df_keys.sort_values(by = ['date','symbol'])
    df_keys = df_keys.dropna()
    df_keys.to_parquet('factor/{}.parq'.format(factor_name))

def setup_logging(filename, filemode):
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    logger = logging.getLogger()
    logger.handlers.clear()
    if local_rank == 0:
        logging.basicConfig(level=logging.INFO, 
                            filename=filename, 
                            filemode=filemode,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=100)
    
    return logger

def select_random_percentage(lst, percentage):
    """
    从列表lst中随机选取指定百分比的元素。
    
    :param lst: 原始列表
    :param percentage: 需要抽取的百分比（例如，0.08表示8%）
    :return: 抽取的元素列表
    """
    k = max(1, int(len(lst) * percentage))  # 计算要抽取的元素数量，至少为1
    return random.sample(lst, k)

def ratio_bars(bars):
    bars['num'] = (bars['large_trade_num'] + bars['med_trade_num'] + bars['small_trade_num']+1e-10)
    bars['vol'] = (bars['large_trade_vol'] + bars['med_trade_vol'] + bars['small_trade_vol']+1e-10)
    bars['val'] = (bars['large_trade_val'] + bars['med_trade_val'] + bars['small_trade_val']+1e-10)
    bars['init_val'] = (bars['init_buy_trade_val'] + bars['init_sell_trade_val']+1e-10)
    bars['init_vol'] = (bars['init_buy_trade_vol'] + bars['init_sell_trade_vol']+1e-10)
    bars['init_num'] = (bars['init_buy_trade_num'] + bars['init_sell_trade_num']+1e-10)
    bars['large_trade_num'] = bars['large_trade_num'] / bars['num']
    bars['small_trade_num'] = bars['small_trade_num'] / bars['num']
    bars['med_trade_num'] = bars['med_trade_num'] / bars['num']
    bars['large_trade_vol'] = bars['large_trade_vol'] / bars['vol']
    bars['small_trade_vol'] = bars['small_trade_vol'] / bars['vol']
    bars['med_trade_vol'] = bars['med_trade_vol'] / bars['vol']
    bars['large_trade_val'] = bars['large_trade_val'] / bars['val']
    bars['small_trade_val'] = bars['small_trade_val'] / bars['val']
    bars['med_trade_val'] = bars['med_trade_val'] / bars['val']
    bars['init_buy_trade_val'] = bars['init_buy_trade_val'] / bars['init_val']
    bars['init_sell_trade_val'] = bars['init_sell_trade_val'] / bars['init_val']
    bars['init_buy_trade_vol'] = bars['init_buy_trade_vol'] / bars['init_vol']
    bars['init_sell_trade_vol'] = bars['init_sell_trade_vol'] / bars['init_vol']
    bars['init_buy_trade_num'] = bars['init_buy_trade_num'] / bars['init_num']
    bars['init_sell_trade_num'] = bars['init_sell_trade_num'] / bars['init_num']

    bars = bars.drop(['num','vol','val','init_val','init_vol','init_num'], axis=1)
    return bars

def remove_outliers_mad(df, threshold=3):
    df_copy = df.copy()
    for column in df_copy.columns:
        if column not in ['date', 'symbol', 'time']:
            median = df_copy[column].median()
            mad = np.median(np.abs(df_copy[column] - median))
            upper_limit = median + threshold * mad
            lower_limit = median - threshold * mad
            df_copy[column] = np.where(df_copy[column] > upper_limit, upper_limit, df_copy[column])
            df_copy[column] = np.where(df_copy[column] < lower_limit, lower_limit, df_copy[column])

    return df_copy