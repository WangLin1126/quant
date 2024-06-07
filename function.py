from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import pandas as pd
import torch

def split(data_array, ret_array, date, window):
    num_rows = data_array.shape[0]
    if num_rows >= window + 4:  
        windowed_data = sliding_window_view(data_array, window_shape=(window, data_array.shape[1]))
        targets = ret_array[window + 3:num_rows]  
        date_array = date[window + 3:num_rows]
        return windowed_data[:len(targets)], targets, date_array
    return np.empty((0, window, data_array.shape[1])), np.array([]), np.array([])

def standardize_and_split(df, prediction_size, pools_df, window):
    df['ret_predict'] = df.groupby('symbol')['ret'].transform(
        lambda x: x.rolling(window=prediction_size).apply(lambda y: (y + 1).prod(), raw=True) - 1
    )
    df = df.dropna()
    filtered_df = pd.merge(df, pools_df, on=['date', 'symbol'], how='inner')

    columns = filtered_df.columns.difference(['date', 'symbol'])
    standardized_df = filtered_df.copy()  
    standardized_df[columns] = (
        filtered_df[columns]
        .groupby(filtered_df['date'])
        .transform(lambda x: (x - x.mean()) / x.std())
    )

    groups = {symbol: group for symbol, group in standardized_df.groupby('symbol')}
    columns = filtered_df.columns.difference(['date', 'symbol', 'ret_predict'])
    time_series = []
    targets = []
    symbols = []
    dates = []
    for symbol, group_data in groups.items():
        data_array = group_data[columns].values
        ret_array = group_data['ret_predict'].values
        date = group_data['date'].values
        result, y_values, date_values = split(data_array, ret_array, date, window)
        if result.size > 0:
            time_series.append(result)
            targets.append(y_values)
            dates.append(date_values)
            repeated_symbols = np.repeat(symbol, date_values.shape[0])
            symbols.append(repeated_symbols)

    data_list = [np.squeeze(np.array(arr),axis=1) for arr in time_series]  
    stacked_data = np.vstack(data_list)  
    X = torch.tensor(stacked_data, dtype=torch.float32)

    stacked_targets = np.concatenate(targets)
    y = torch.tensor(stacked_targets, dtype=torch.float32).view(-1, 1)

    return X , y , dates , symbols
