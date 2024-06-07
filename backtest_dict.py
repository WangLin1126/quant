import pickle
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
from datasets.dataset import *

# test_begin_date = '2015-01-03'
test_begin_date = '2019-06-01'
test_end_date = '2020-06-30'
threshold_of_grow_and_decline = 0.5
window = 30
prediction_size = 5
input_size = 6
if input_size == 6:
    adds = False
else:
    adds = True
norm = True
dataset , df_keys = time_series_dataset_dict_cross(
                train_begin_date = test_begin_date,  test_end_date = test_end_date,
                threshold_of_grow_and_decline = threshold_of_grow_and_decline, 
                prediction_size = prediction_size, adds = adds,
                window = window, norm = norm)
test_dataset = {}
for timestamp, data in dataset.items():
    if pd.Timestamp(test_begin_date) <= timestamp <= pd.Timestamp(test_end_date):
        test_dataset[timestamp] = data

test_keys = df_keys[df_keys['date']>=pd.Timestamp(test_begin_date)]
test_keys = test_keys.sort_values(by = ['date','symbol'])
test_keys = test_keys.reset_index(drop=True)
output_dim = 1
c = None
# fold = 10
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# hidden_dim = 256
# num_layers = 3
# factor_name = "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}".format(input_size , prediction_size , window , hidden_dim , num_layers)
# backtest = None

# for i in range(fold):
#     model_name = './models/input_size_{}_prediction_size_10_window_30_hidden_dim_256_num_layers_3_fold_{}_val.pth'.format(input_size , i)
#     model = LSTM(input_size, hidden_dim, num_layers, output_dim).to(device)
#     checkpoint = torch.load(model_name, map_location = device)
#     model.load_state_dict(checkpoint['model'])
#     model.eval()
#     with torch.no_grad():
#         outputs = []
#         keys = list(test_dataset.keys())
#         for key in keys:
#             data, target = test_dataset[key].tensors[0] , test_dataset[key].tensors[1]
#             data, target = data.to(device), target.to(device)
#             target = target.view(-1)
#             outputs.append(model(data))
#         if backtest is None:
#             backtest = torch.cat(outputs).cpu().numpy()
#         else:
#             backtest += torch.cat(outputs).cpu().numpy()
# test_keys[factor_name] = backtest
# test_keys.to_parquet('factor/{}.parq'.format(factor_name))    
# factor_dir = os.path.join(os.path.expanduser('~'), 'backtest/factors', factor_name, '1000')
# if not os.path.exists(factor_dir):
#     os.makedirs(factor_dir)

# data_file = os.path.join(os.getcwd(), 'factor', f'{factor_name}.parq')
# destination_file = os.path.join(factor_dir, 'data.parq')
# if os.path.isfile(data_file):
#     shutil.copyfile(data_file, destination_file)
#     print ("backtest data generate end: {}".format(factor_name))
# else:
#     print(f"data {data_file} does not exist.")    

model_dir = 'models/'
for model_name in os.listdir(model_dir):
    if model_name.endswith('train.pth'):
        # match = re.search(r'cross_Truenorm_train_begin_date_2015-01-03_c_(\d+)_input_size_(\d+)_prediction_size_(\d+)_window_(\d+)_hidden_dim_(\d+)_num_layers_(\d+)_IC', model_name)
        match = re.search(r'gruroll_Truenorm_train_begin_date_2015-01-03_input_size_(\d+)_prediction_size_(\d+)_window_(\d+)_hidden_dim_(\d+)_num_layers_(\d+)_IC', model_name)
        if (match and int(match.group(1))==input_size and int(match.group(2))==prediction_size):
            print ("backtest data generate begin: {}".format(model_name))
            # c = int(match.group(1))
            window = int(match.group(3))
            hidden_dim = int(match.group(4))
            num_layers = int(match.group(5))

            factor_name = match.group(0)
            backtest_data_generate_dict(
                    df_keys = test_keys.copy(),dataset = test_dataset.copy(),
                    input_size=input_size, output_dim=output_dim,
                    window=window,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    model_name= model_dir + model_name,
                    factor_name=factor_name,
                    device=device,
                    c = c)

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