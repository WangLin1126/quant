import fds
import pandas as pd 
from datasets.function import standardize_and_split
from utilities import *
from typing import Union
import pickle
def process_raw_df(train_begin_date = '2015-01-03', train_end_date = '2018-12-31', 
                   valid_begin_date = '2019-01-03', valid_end_date = '2019-12-31',
                   test_begin_date = '2020-01-03',test_end_date = '2020-12-31',
                   threshold_of_grow_and_decline = 0.5, prediction_size = 3,
                   window = 36):

    status = fds.list_status(train_begin_date, test_end_date, columns=None)
    bars = fds.bar(train_begin_date, test_end_date, freq='1d')
    # 剔除 涨跌停超过一定值 PT ST 股票池

    flager = fds.bar(train_begin_date, test_end_date, freq='5m')
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
    filtered = merged[(merged['flag'] != 1)]
    # filtered = merged[(merged['PT'] != 1) & (merged['ST'] != 1) & (merged['turnover'] != 0) & (merged['flag'] != 1)]
    pools = {date: symbols.reset_index(drop=True) for date, symbols in filtered.groupby('date')['symbol']}
    pools_df = pd.concat(pools.values(), keys=pools.keys()).reset_index().drop('level_1', axis=1)
    pools_df.columns = ['date', 'symbol']

    # 添加预测日收益率 标准化 数据集拆分 添加因子
    df = bars
    df['ret'] = df['close']/df['pre_close'] - 1
    df = add_factors(df) #####################添加因子

    train_df = df[df['date'] <= pd.Timestamp(train_end_date)]
    valid_df = df[(df['date'] >= pd.Timestamp(valid_begin_date)) & (df['date'] <= pd.Timestamp(valid_end_date))]
    test_df = df[df['date'] >= pd.Timestamp(test_begin_date)]

    train_X , train_y , _ , _ = standardize_and_split(train_df, prediction_size, pools_df, window)
    valid_X , valid_y , _ , _ = standardize_and_split(valid_df, prediction_size, pools_df, window)
    test_X , test_y , _ , _ = standardize_and_split(test_df, prediction_size, pools_df, window)

    return train_X, train_y, valid_X ,valid_y , test_X , test_y

def generate_pools(train_begin_date : Union[pd.Timestamp, str],test_end_date : Union[pd.Timestamp, str],
                    threshold_of_grow_and_decline = 0.5 ):
    """
    返回一个字典，一个dataframe
    """
    status = fds.list_status(train_begin_date, test_end_date, columns=None)
    bars = fds.bar(train_begin_date, test_end_date, freq='1d')
    # 涨跌停筛选 截面日超过一定时间比例则删 flag为1 
    flager = fds.bar(train_begin_date, test_end_date, freq='5m')
    high_max = flager.groupby(['date', 'symbol'])['high'].transform('max')
    low_min = flager.groupby(['date', 'symbol'])['low'].transform('min')
    flager['close_ge_high'] = (flager['close'] >= high_max).astype(int )
    flager['close_le_low'] = (flager['close'] <= low_min).astype(int)
    flags_high_ratio = flager.groupby(['date', 'symbol'])['close_ge_high'].transform('mean')
    flags_low_ratio = flager.groupby(['date', 'symbol'])['close_le_low'].transform('mean')
    flager['flag'] = ((flags_high_ratio > threshold_of_grow_and_decline) | (flags_low_ratio > threshold_of_grow_and_decline)).astype(int)
    daily_flags = (flager.groupby(['date', 'symbol'])['flag']
                .first()
                .reset_index())
    pools = {}
    # PT当日无分钟频数据（PT）删除 上市日期小于等于10 每日流通量最低的5%  
    # 每日成交比最少的4%股票PT大概在4.5% 最终删除率在10%左右
    merged = pd.merge(bars, daily_flags, on=['date', 'symbol'], how='left')
    merged['flag'] = merged['flag'].fillna(1)
    merged = pd.merge(merged, status, on=['date', 'symbol'], how='left')
    merged[['ST','PT']] = merged[['ST','PT']].fillna(0)
    merged['listed_days'] = merged['listed_days'].fillna(1)
    merged['flag'] = merged['flag'].astype(int) | (merged['listed_days'] <= 10).astype(int)
    merged['rank'] = merged.groupby('date')['match_items'].rank(pct=True)
    merged.loc[merged['rank'] <= 0.05, 'flag'] = 1

    # 股票池
    filtered = merged[(merged['flag'] != 1)]
    pools = {date: symbols.reset_index(drop=True) for date, symbols in filtered.groupby('date')['symbol']}
    pools_df = pd.concat(pools.values(), keys=pools.keys()).reset_index().drop('level_1', axis=1)
    pools_df.columns = ['date', 'symbol']
    return pools , pools_df

def process_date(df, pools, window, shifted_dates, date): 
    if date not in pools:
        return None
    valid_symbols = pools[date]
    mask = (df['symbol'].isin(valid_symbols)) & (df['date'] <= date) & (df['date'] > shifted_dates[date])
    date_data = df[mask]
    if len(date_data)<window:
        return {} 
    date_data = date_data.sort_values(by = ['symbol','date'])
    date_data = date_data.reset_index(drop=True)
    all_data_tensor = torch.tensor(date_data.drop(['date', 'symbol'], axis=1).values, dtype=torch.float32)
    symbols = date_data['symbol'].values
    unique_symbols, counts = np.unique(symbols, return_counts=True)
    tensors = dict(zip(unique_symbols, torch.split(all_data_tensor, counts.tolist())))
    tensors_dict = {symbol: tensor for symbol, tensor in tensors.items() if tensor.shape[0] >= window}
    return tensors_dict 

def process_date_minutes(df, pools, window, shifted_dates, date,date_data): 
    if date not in pools:
        return None
    valid_symbols = pools[date]
    mask = (df['symbol'].isin(valid_symbols)) & (df['date'] <= date) & (df['date'] > shifted_dates[date])
    data = df[mask]
    if len(data)<window*date_data:
        return {} 
    data = data.sort_values(by = ['symbol','date','time'])
    data = data.reset_index(drop=True)
    all_data_tensor = torch.tensor(data.drop(['date', 'symbol','time'], axis=1).values, dtype=torch.float32)
    symbols = data['symbol'].values
    unique_symbols, counts = np.unique(symbols, return_counts=True)
    tensors = dict(zip(unique_symbols, torch.split(all_data_tensor, counts.tolist())))
    tensors_dict = {symbol: tensor for symbol, tensor in tensors.items() if tensor.shape[0] >= window*date_data}
    return tensors_dict 

def time_series_dataset_dict(begin_date : Union[pd.Timestamp, str],  end_date : Union[pd.Timestamp, str],
                            prediction_size = 10, window = 30, adds = False, drop = False, 
                            norm = True,freq = '1d', ex = False, outstand = False, drop0930 = False, ratio = False, winsor = True):
    with open('./datasets/pools.pkl', 'rb') as f:
        pools = pickle.load(f)

    if freq == '1d':
        if ex:
            bars = fds.exbar_mini(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq=freq)
            bars = bars[['date','symbol','open','close','high','low','volume','turnover','trade_num', #高开低收量额 ，买卖， 大小单
                         'init_buy_trade_num','init_buy_trade_vol','init_buy_trade_val','init_sell_trade_num','init_sell_trade_vol','init_sell_trade_val',
                         'large_trade_num','large_trade_vol','large_trade_val','med_trade_num','med_trade_vol','med_trade_val','small_trade_num','small_trade_vol','small_trade_val']]
            if ratio:
                bars=ratio_bars(bars)
        else: 
            bars = fds.bar(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq=freq)
        
        if outstand:
            dataframe = pd.read_parquet('/root/gru/datasets/outstand.parq')
            cols_to_fill = dataframe.columns[dataframe.columns != 'symbol']
            dataframe[cols_to_fill] = dataframe.groupby('date')[cols_to_fill].transform(lambda x: x.fillna(x.mean()))
            dataframe = dataframe.reset_index(drop = True)
            df = pd.merge(bars, dataframe, how='left', on = ['date','symbol'])
        else:
            df = bars.copy()
    else:
        if ex:
            bars1 = fds.exbar_mini(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq='1d')
            bars1 = bars1[['date','symbol','open','close','high','low','volume','turnover','trade_num', #高开低收量额 ，买卖， 大小单
                'init_buy_trade_num','init_buy_trade_vol','init_buy_trade_val','init_sell_trade_num','init_sell_trade_vol','init_sell_trade_val',
                'large_trade_num','large_trade_vol','large_trade_val','med_trade_num','med_trade_vol','med_trade_val','small_trade_num','small_trade_vol','small_trade_val']]
            bars2 = fds.exbar_mini(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq=freq)
            bars2 = bars2[['date','symbol','time','open','close','high','low','volume','turnover','trade_num', #高开低收量额 ，买卖， 大小单
                'init_buy_trade_num','init_buy_trade_vol','init_buy_trade_val','init_sell_trade_num','init_sell_trade_vol','init_sell_trade_val',
                'large_trade_num','large_trade_vol','large_trade_val','med_trade_num','med_trade_vol','med_trade_val','small_trade_num','small_trade_vol','small_trade_val']]
            if ratio:
                bars1=ratio_bars(bars1)
                bars2=ratio_bars(bars2)
        else:
            bars1 = fds.bar(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq='1d')
            bars2 = fds.bar(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq=freq)
        if outstand:
            dataframe = pd.read_parquet('/root/gru/datasets/outstand.parq')
            cols_to_fill = dataframe.columns[dataframe.columns != 'symbol']
            dataframe[cols_to_fill] = dataframe.groupby('date')[cols_to_fill].transform(lambda x: x.fillna(x.mean()))
            dataframe = dataframe.reset_index(drop = True)
            bars1 = pd.merge(bars1, dataframe, how='left', on = ['date','symbol'])
        if drop0930:
            bars2 = bars2[bars2['time'] != '0 days 09:30:00']
        time_stamps = bars2['time'].unique()
        time_df = pd.DataFrame({'time': time_stamps})
        repeat_times = len(time_stamps)
        dfnew = pd.DataFrame(np.repeat(bars1.values,repeat_times,axis=0))
        dfnew.columns = bars1.columns
        group_size = len(dfnew) // len(time_df)
        time_df_expanded = pd.concat([time_df]*group_size, ignore_index=True)
        dfnew = pd.concat([dfnew, time_df_expanded], axis=1)
        dfnew = dfnew.sort_values(by = ['date','symbol']).reset_index(drop=True)
        merged_df = pd.merge(dfnew, bars2, on=['date', 'symbol','time'], how='left', suffixes=('_df1', '_df2'))
        columns_to_update = bars2.columns.difference(['date', 'symbol','time'])
        for col in columns_to_update:
            merged_df[col] = merged_df[f'{col}_df2'].fillna(merged_df[f'{col}_df1'])
        merged_df = merged_df[['date','symbol','time'] + columns_to_update.tolist()]
        df = merged_df.copy()
    ### 去极值
    if winsor:
        df = remove_outliers_mad(df)
    adjusting_factor = fds.adjusting_factor(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1))
    df['date'] = pd.to_datetime(df['date'])
    adjusting_factor['date'] = pd.to_datetime(adjusting_factor['date'])
    mg = pd.merge(df, adjusting_factor ,how= 'outer')
    mg['ratio_adjusting_factor'].fillna(1, inplace=True)
    mg['close'] = mg['close'] * mg['ratio_adjusting_factor']
    mg['high'] = mg['high'] * mg['ratio_adjusting_factor']
    mg['low'] = mg['low'] * mg['ratio_adjusting_factor']
    mg['open'] = mg['open'] * mg['ratio_adjusting_factor']
    mg['volume'] = mg['volume'] / (mg['ratio_adjusting_factor']+1e-6)
    if ex:
        mg = mg.drop(['adjusting_factor','adjusting_const','ratio_adjusting_factor'], axis=1)
    else:
        mg = mg.drop(['adjusting_factor','adjusting_const','ratio_adjusting_factor','pre_close','match_items'], axis=1)
    mg = mg.dropna()
    if freq == '1d':
        date_data = 1
    else:
        date_data = mg.groupby(['date', 'symbol'])['time'].nunique().max()
    mg['open_shift1'] = mg.groupby('symbol')['open'].shift(-1*date_data)
    mg['open_shiftn'] = mg.groupby('symbol')['open'].shift(-(prediction_size + 1)*date_data)
    mg['ret_predict'] = mg['open_shiftn']/(mg['open_shift1']+1e-6) -1
    mg = mg.drop(['open_shiftn','open_shift1'], axis=1)
    mg = mg.dropna()
    df = mg.copy()
    if adds:
        df = add_factors(df)
    if drop:
        columns = df.columns.difference(['high','low', 'open','close','volume','turnover'])
        df = df[columns]
    
    df = df.dropna()    
    #ret_predict放到最后一列
    ret_column = df.pop('ret_predict')
    df['ret_predict'] = ret_column
    df = df.reset_index(drop=True)

    unique_dates = df['date'].unique()
    shifted_dates = {date: fds.shift_trading_day(date, -window) for date in unique_dates}
    dic = {}
    dataset = {}
    for date in df['date'].unique():
        if freq == '1d':
            tensors_dict = process_date(df=df,pools=pools,window=window,shifted_dates=shifted_dates,date=date) 
        else:
            tensors_dict = process_date_minutes(df=df,pools=pools,window=window,shifted_dates=shifted_dates,date=date,date_data=date_data)  
        if len(tensors_dict) != 0  :
            dic[date] = tensors_dict
    df_list = [pd.DataFrame({'date': outer_key, 'symbol': list(inner_dict.keys())})
           for outer_key, inner_dict in dic.items()]
    df_keys = pd.concat(df_list, ignore_index=True)
    for key in dic.keys():
        day_tensor = torch.stack([tensor for tensor in dic[key].values()],dim=0)
        y = day_tensor[:,-date_data,-1]
        X = day_tensor[:,:,0:-1]
        if norm:
            # X时序标准化
            X = X / (X[: , -1 , :].unsqueeze(1)+1e-6)
            # X = (X - mean_seq) / (std_seq + 1e-8)
            # X截面标准化
            mean_cross = X.mean(dim=0, keepdim=True)  
            std_cross = X.std(dim=0, keepdim=True)    
            X = (X - mean_cross) / (std_cross + 1e-6) 
            y  = (y-y.mean())/(y.std()+1e-6) 
        X.requires_grad_(True)
        y.requires_grad_(True)
        dataset[key] = TensorDataset(X,y)
    return dataset , df_keys

def time_series_dataset_dict_cross(begin_date : Union[pd.Timestamp, str],  end_date : Union[pd.Timestamp, str],
                            threshold_of_grow_and_decline = 0.5, prediction_size = 10,
                            window = 30, adds = False, drop = False, norm = True,freq = '1d', ex = False, outstand = False, 
                            drop0930 = False):
    
    with open('./datasets/pools.pkl', 'rb') as f:
        pools = pickle.load(f)

    if freq == '1d':
        if ex:
            bars = fds.exbar_mini(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq=freq)
        else: 
            bars = fds.bar(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq=freq)
        
        if outstand:
            dataframe = pd.read_parquet('/root/gru/datasets/outstand.parq')
            cols_to_fill = dataframe.columns[dataframe.columns != 'symbol']
            dataframe[cols_to_fill] = dataframe.groupby('date')[cols_to_fill].transform(lambda x: x.fillna(x.mean()))
            dataframe = dataframe.reset_index(drop = True)
            df = pd.merge(bars, dataframe, how='left', on = ['date','symbol'])
        else:
            df = bars.copy()
    else:
        if ex:
            bars1 = fds.exbar_mini(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq='1d')
            bars2 = fds.exbar_mini(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq=freq)
        else:
            bars1 = fds.bar(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq='1d')
            bars2 = fds.bar(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq=freq)
        if outstand:
            dataframe = pd.read_parquet('/root/gru/datasets/outstand.parq')
            cols_to_fill = dataframe.columns[dataframe.columns != 'symbol']
            dataframe[cols_to_fill] = dataframe.groupby('date')[cols_to_fill].transform(lambda x: x.fillna(x.mean()))
            dataframe = dataframe.reset_index(drop = True)
            bars1 = pd.merge(bars1, dataframe, how='left', on = ['date','symbol'])
        if drop0930:
            bars2 = bars2[bars2['time'] != '0 days 09:30:00']
        time_stamps = bars2['time'].unique()
        time_df = pd.DataFrame({'time': time_stamps})
        repeat_times = len(time_stamps)
        dfnew = pd.DataFrame(np.repeat(bars1.values,repeat_times,axis=0))
        dfnew.columns = bars1.columns
        group_size = len(dfnew) // len(time_df)
        time_df_expanded = pd.concat([time_df]*group_size, ignore_index=True)
        dfnew = pd.concat([dfnew, time_df_expanded], axis=1)
        dfnew = dfnew.sort_values(by = ['date','symbol']).reset_index(drop=True)
        merged_df = pd.merge(dfnew, bars2, on=['date', 'symbol','time'], how='left', suffixes=('_df1', '_df2'))
        columns_to_update = bars2.columns.difference(['date', 'symbol','time'])
        for col in columns_to_update:
            merged_df[col] = merged_df[f'{col}_df2'].fillna(merged_df[f'{col}_df1'])
        merged_df = merged_df[['date','symbol','time'] + columns_to_update.tolist()]
        df = merged_df.copy()

    adjusting_factor = fds.adjusting_factor(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1))
    mg = pd.merge(df, adjusting_factor ,how= 'outer')
    mg['ratio_adjusting_factor'].fillna(1, inplace=True)
    mg['close'] = mg['close'] * mg['ratio_adjusting_factor']
    mg['high'] = mg['high'] * mg['ratio_adjusting_factor']
    mg['low'] = mg['low'] * mg['ratio_adjusting_factor']
    mg['open'] = mg['open'] * mg['ratio_adjusting_factor']
    mg['volume'] = mg['volume'] / (mg['ratio_adjusting_factor']+1e-8)
    if ex:
        mg = mg.drop(['adjusting_factor','adjusting_const','ratio_adjusting_factor','pred_close','trade_vol','trade_val','preclose'], axis=1)
    else:
        mg = mg.drop(['adjusting_factor','adjusting_const','ratio_adjusting_factor','pre_close','match_items'], axis=1)
    mg = mg.dropna()
    if freq == '1d':
        date_data = 1
    else:
        date_data = mg.groupby(['date', 'symbol'])['time'].nunique().max()
    mg['open_shift1'] = mg.groupby('symbol')['open'].shift(-1*date_data)
    mg['open_shiftn'] = mg.groupby('symbol')['open'].shift(-(prediction_size + 1)*date_data)
    mg['ret_predict'] = mg['open_shiftn']/(mg['open_shift1']+1e-8) -1
    mg = mg.drop(['open_shiftn','open_shift1'], axis=1)
    mg = mg.dropna()
    df = mg.copy()
    if adds:
        df = add_factors(df)
    if drop:
        columns = df.columns.difference(['high','low', 'open','close','volume','turnover'])
        df = df[columns]
    df = df.dropna()    
    #ret_predict放到最后一列
    ret_column = df.pop('ret_predict')
    df['ret_predict'] = ret_column
    df = df.reset_index(drop=True)

    unique_dates = df['date'].unique()
    shifted_dates = {date: fds.shift_trading_day(date, -window) for date in unique_dates}
    dic = {}
    dataset = {}
    for date in df['date'].unique():
        if freq == '1d':
            tensors_dict = process_date(df=df,pools=pools,window=window,shifted_dates=shifted_dates,date=date) 
        else:
            tensors_dict = process_date_minutes(df=df,pools=pools,window=window,shifted_dates=shifted_dates,date=date,date_data=date_data)  
        if len(tensors_dict) != 0  :
            dic[date] = tensors_dict
    df_list = [pd.DataFrame({'date': outer_key, 'symbol': list(inner_dict.keys())})
           for outer_key, inner_dict in dic.items()]
    df_keys = pd.concat(df_list, ignore_index=True)
    for key in dic.keys():
        day_tensor = torch.stack([tensor for tensor in dic[key].values()],dim=0)
        y = day_tensor[:,-date_data,-1]
        X = day_tensor[:,:,0:-1]
        if norm:
            # X时序标准化
            X = X / (X[: , -1 , :].unsqueeze(1)+1e-8)
            # X = (X - mean_seq) / (std_seq + 1e-8)
            # X截面标准化
            mean_cross = X.mean(dim=0, keepdim=True)  
            std_cross = X.std(dim=0, keepdim=True)    
            X = (X - mean_cross) / (std_cross + 1e-8) 
            y  = (y-y.mean())/(y.std()+1e-8) 
        X.requires_grad_(True)
        y.requires_grad_(True)
        dataset[key] = TensorDataset(X,y)
    return dataset , df_keys

def time_series_dataset_dict_cross1(begin_date : Union[pd.Timestamp, str],  end_date : Union[pd.Timestamp, str],
                            threshold_of_grow_and_decline = 0.5, prediction_size = 10,
                            window = 30, adds = False, drop = False, norm = True,freq = '1d', ex = False, outstand = False, 
                            drop0930 = False):
    
    with open('./datasets/pools.pkl', 'rb') as f:
        pools = pickle.load(f)

    if freq == '1d':
        if ex:
            bars = fds.exbar_mini(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq=freq)
        else: 
            bars = fds.bar(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq=freq)
        
        if outstand:
            dataframe = pd.read_parquet('/root/gru/datasets/outstand.parq')
            cols_to_fill = dataframe.columns[dataframe.columns != 'symbol']
            dataframe[cols_to_fill] = dataframe.groupby('date')[cols_to_fill].transform(lambda x: x.fillna(x.mean()))
            dataframe = dataframe.reset_index(drop = True)
            df = pd.merge(bars, dataframe, how='left', on = ['date','symbol'])
        else:
            df = bars.copy()
    else:
        if ex:
            bars1 = fds.exbar_mini(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq='1d')
            bars2 = fds.exbar_mini(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq=freq)
        else:
            bars1 = fds.bar(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq='1d')
            bars2 = fds.bar(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1), freq=freq)
        if outstand:
            dataframe = pd.read_parquet('../factor_shared/factors_outstanding_0429.parq')
            dataframe = dataframe.groupby('symbol').apply(lambda group: group.ffill())
            dataframe = dataframe.reset_index(drop = True)
            bars1 = pd.merge(bars1, dataframe, how='left', on = ['date','symbol'])
        if drop0930:
            bars2 = bars2[bars2['time'] != '0 days 09:30:00']
        time_stamps = bars2['time'].unique()
        time_df = pd.DataFrame({'time': time_stamps})
        repeat_times = len(time_stamps)
        dfnew = pd.DataFrame(np.repeat(bars1.values,repeat_times,axis=0))
        dfnew.columns = bars1.columns
        group_size = len(dfnew) // len(time_df)
        time_df_expanded = pd.concat([time_df]*group_size, ignore_index=True)
        dfnew = pd.concat([dfnew, time_df_expanded], axis=1)
        dfnew = dfnew.sort_values(by = ['date','symbol']).reset_index(drop=True)
        merged_df = pd.merge(dfnew, bars2, on=['date', 'symbol','time'], how='left', suffixes=('_df1', '_df2'))
        columns_to_update = bars2.columns.difference(['date', 'symbol','time'])
        for col in columns_to_update:
            merged_df[col] = merged_df[f'{col}_df2'].fillna(merged_df[f'{col}_df1'])
        merged_df = merged_df[['date','symbol','time'] + columns_to_update.tolist()]
        df = merged_df.copy()

    adjusting_factor = fds.adjusting_factor(fds.shift_trading_day(begin_date, -window-1), fds.shift_trading_day(end_date, prediction_size + 1))
    mg = pd.merge(df, adjusting_factor ,how= 'outer')
    mg['ratio_adjusting_factor'].fillna(1, inplace=True)
    mg['close'] = mg['close'] * mg['ratio_adjusting_factor']
    mg['high'] = mg['high'] * mg['ratio_adjusting_factor']
    mg['low'] = mg['low'] * mg['ratio_adjusting_factor']
    mg['open'] = mg['open'] * mg['ratio_adjusting_factor']
    mg['volume'] = mg['volume'] / (mg['ratio_adjusting_factor']+1e-8)
    if ex:
        mg = mg.drop(['adjusting_factor','adjusting_const','ratio_adjusting_factor','pred_close','trade_vol','trade_val','preclose'], axis=1)
    else:
        mg = mg.drop(['adjusting_factor','adjusting_const','ratio_adjusting_factor','pre_close','match_items'], axis=1)
    mg = mg.dropna()
    if freq == '1d':
        date_data = 1
    else:
        date_data = mg.groupby(['date', 'symbol'])['time'].nunique().max()
    mg['open_shift1'] = mg.groupby('symbol')['open'].shift(-1*date_data)
    mg['open_shiftn'] = mg.groupby('symbol')['open'].shift(-(prediction_size + 1)*date_data)
    mg['ret_predict'] = mg['open_shiftn']/(mg['open_shift1']+1e-8) -1
    mg = mg.drop(['open_shiftn','open_shift1'], axis=1)
    mg = mg.dropna()
    df = mg.copy()
    if adds:
        df = add_factors(df)
    if drop:
        columns = df.columns.difference(['high','low', 'open','close','volume','turnover'])
        df = df[columns]
    df = df.dropna()    
    #ret_predict放到最后一列
    ret_column = df.pop('ret_predict')
    df['ret_predict'] = ret_column
    df = df.reset_index(drop=True)

    unique_dates = df['date'].unique()
    shifted_dates = {date: fds.shift_trading_day(date, -window) for date in unique_dates}
    dic = {}
    dataset = {}
    for date in df['date'].unique():
        if freq == '1d':
            tensors_dict = process_date(df=df,pools=pools,window=window,shifted_dates=shifted_dates,date=date) 
        else:
            tensors_dict = process_date_minutes(df=df,pools=pools,window=window,shifted_dates=shifted_dates,date=date,date_data=date_data)  
        if len(tensors_dict) != 0  :
            dic[date] = tensors_dict
    df_list = [pd.DataFrame({'date': outer_key, 'symbol': list(inner_dict.keys())})
           for outer_key, inner_dict in dic.items()]
    df_keys = pd.concat(df_list, ignore_index=True)
    for key in dic.keys():
        day_tensor = torch.stack([tensor for tensor in dic[key].values()],dim=0)
        y = day_tensor[:,-date_data,-1]
        X = day_tensor[:,:,0:-1]
        if norm:
            # X时序标准化
            X = X / (X[: , -1 , :].unsqueeze(1)+1e-8)
            # X = (X - mean_seq) / (std_seq + 1e-8)
            # X截面标准化
            mean_cross = X.mean(dim=0, keepdim=True)  
            std_cross = X.std(dim=0, keepdim=True)    
            X = (X - mean_cross) / (std_cross + 1e-8) 
            y  = (y-y.mean())/(y.std()+1e-8) 
        # X.requires_grad_(True)
        # y.requires_grad_(True)
        dataset[key] = TensorDataset(X,y)
    return dataset , df_keys