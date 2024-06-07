import torch
import pandas as pd
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset , TensorDataset
from sklearn.model_selection import ParameterGrid
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import *
import os
import logging
from model import LSTM
from datasets.dataset import *
from engine import *
from loss import *
import shutil
pd.options.mode.chained_assignment = None

def main(test_flag=False, model_dir="models/"):
    train_begin_date = '2015-01-03'
    train_end_date = '2018-12-31' 
    test_begin_date = '2019-07-01'
    test_end_date = '2020-06-30'
    adds = False
    norm = True
    drop0930 = False
    input_size = 20
    input_sizes = [6,6]
    freqs =  ['1d','30m']
    hidden_dim = 64
    output_dim = 1
    threshold_of_grow_and_decline = 0.5
    fold = 10
    dropout_rate = 0.2
    c = 30
    num_heads = 2
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    prediction_size = 25
    window = 50
    num_layers = 3
    
    # factor_name = "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}".format(input_size , prediction_size , window , hidden_dim , num_layers)
    model_name = "./models/cross_Truenorm_train_begin_date_2015-01-03_c_30_input_sizes_[6, 6]_prediction_size_25_window_50_hidden_dim_64_num_layers_3_independent_['1d', '30m']_1year_train.pth"

    # model = LSTM(input_size, hidden_dim, num_layers, output_dim).to(device)
    model = MultiGRU_Independent(input_sizes, hidden_dim, num_layers, output_dim, dropout_rate, c, num_heads, device).to(device)
    checkpoint = torch.load(model_name, map_location = device)
    model.load_state_dict(checkpoint['model'])
    print('Load Model Success!')
    print('Start Get Test Dataset!')
    
    for i in range (input_sizes[0]):
        factor_name = f"del{i}_cross_Truenorm_train_begin_date_2015-01-03_c_30_input_sizes_[6, 6]_prediction_size_25_window_50_hidden_dim_64_num_layers_3_independent_['1d', '30m']_1year_train_train"
        datasets = []
        test_datasets = []
        df_keys = []
        test_keys = []
        for index in range(len(freqs)):
            drop = False
            outstand = False
            ex = False
            adds = False
            input_size = input_sizes[index]
            if input_size == 54:
                adds = True
                ex = True
            elif input_size == 40:
                ex = True
            elif input_size == 20:
                adds = True
            elif input_size == 14:
                adds = True
                drop = True
            elif input_size == 44:
                outstand = True
                drop = True
            else:
                pass
            freq = freqs[index]
            dataset , df_key = time_series_dataset_dict_cross1(
                        begin_date = test_begin_date,  end_date = test_end_date,
                        threshold_of_grow_and_decline = threshold_of_grow_and_decline, 
                        prediction_size = prediction_size, adds = adds, window = window, 
                        drop = drop, norm = norm , freq = freq, ex = ex, outstand = outstand, drop0930 = drop0930)
            for key, tensor in dataset.items():
                mask = torch.zeros_like(tensor.tensors[0])
                mask[:, :, i] = 1
                tensor.tensors[0][mask.bool()] = 0
                # tensor = tensor * mask
                dataset[key] = tensor
            
            datasets.append(dataset)
            df_keys.append(df_key)
            test_dataset = {}
            for timestamp, data in dataset.items():
                if pd.Timestamp(test_begin_date) <= timestamp <= pd.Timestamp(test_end_date):
                    test_dataset[timestamp] = data  
            test_datasets.append(test_dataset)
            test_key = df_key[df_key['date']>=pd.Timestamp(test_begin_date)]
            test_key = test_key.sort_values(by = ['date','symbol'])
            test_key = test_key.reset_index(drop=True)
            test_keys.append(test_key)

        print('Start Calculate Factors!')
        backtest = pd.DataFrame()
        model.eval()
        with torch.no_grad():
            outputs = [] 
            keys = list(test_datasets[0].keys())
            for key in keys:
                datas = []
                targets = []
                for index in range(len(test_datasets)):
                    data, target = test_datasets[index][key].tensors[0] , test_datasets[index][key].tensors[1]
                    data, target = data.to(device), target.to(device)
                    datas.append(data)
                    targets.append(target)
                outputs.append(model(datas))

        test_keys[0][factor_name] = torch.cat(outputs).cpu().numpy()
        backtest = pd.concat([backtest, test_keys[0]], ignore_index=True)

        backtest.reset_index(drop=True)  
        print ("backtest data generate begin: {}".format(factor_name))
        backtest[factor_name] = backtest.groupby('symbol')[factor_name].shift(1)
        backtest = backtest.reset_index(drop = True)
        backtest = backtest.dropna()
        backtest = backtest.sort_values(by = ['date','symbol'])
        backtest.to_parquet('factor/{}.parq'.format(factor_name))    
        factor_dir = os.path.join(os.path.expanduser('~'), 'backtest/factors', factor_name)
        if not os.path.exists(factor_dir):
            os.makedirs(factor_dir)

        data_file = os.path.join(os.getcwd(), 'factor', f'{factor_name}.parq')
        destination_file = os.path.join(factor_dir, 'data.parq')
        if os.path.isfile(data_file):
            shutil.copyfile(data_file, destination_file)
            print ("backtest data generate end: {}".format(factor_name))
        else:
            print(f"data file {data_file} does not exist.")                 


if __name__ == "__main__":
    main(test_flag=True, model_dir="models/")