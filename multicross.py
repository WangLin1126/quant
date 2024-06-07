import torch
import pandas as pd
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset , TensorDataset
from sklearn.model_selection import ParameterGrid
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from model import *
import os
import logging
from datasets.dataset import *
from engine import *
from loss import *
import re
import shutil
pd.options.mode.chained_assignment = None

def main(test_flag=False, model_dir="models/", run_year=4, 
    train_begin_dates = '2015-01-03',
    train_end_dates = '2018-12-31' ,
    valid_begin_dates = '2019-01-03',
    valid_end_dates = '2019-12-31',
    test_begin_dates = '2020-01-01',
    test_end_dates = '2020-12-31'):
    drop = False
    drop0930 = False
    norm = True
    outstand = False
    ex = False
    adds = False
    loss = 'MSE'
    model_state = 'independent'
    log_dir = 'log/'
    patience = 15
    output_dim = 1
    epochs = 100
    lr = 0.001
    dropout_rate = 0.2
    weight_decay=1e-5
    batch_size = 256  
    num_heads=2
    threshold_of_grow_and_decline = 0.5
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    param_grid = {
            'prediction_size': [25],
            'window': [50],
            'hidden_dim': [64,128],
            'num_layers': [3],
            'c': [30],
            'freqs': [['1d','60m']],
            'input_sizes': [[26, 6]],
        }
    param_list = list(ParameterGrid(param_grid))
    
    for params in param_list:
        backtest = pd.DataFrame()
        prediction_size = params['prediction_size']
        window = params['window']
        hidden_dim = params['hidden_dim']
        num_layers = params['num_layers']
        c = params['c']
        freqs = params['freqs']
        input_sizes = params['input_sizes']
        if len(freqs) != len(input_sizes):
            continue
        factor_name = "multicross_{}norm_c_{}_input_sizes_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}_{}_{}_{}year".format(
            norm, c, input_sizes, prediction_size, window, hidden_dim, num_layers, model_state, freqs, run_year)
        # Log
        log_path = log_dir + "multicross_{}norm_c_{}_input_sizes_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}_{}_{}_{}year.log".format(
            norm, c, input_sizes, prediction_size, window, hidden_dim, num_layers, model_state, freqs, run_year)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                filename=log_path, filemode='a')
        logger = logging.getLogger()
        for year in range(run_year):
            train_begin_date , train_end_date , valid_begin_date , valid_end_date , test_begin_date , test_end_date = fds.shift_days(
                [train_begin_dates, train_end_dates, valid_begin_dates, valid_end_dates, test_begin_dates, test_end_dates], year*365)
            train_model_name = model_dir + "multicross_{}norm_train_begin_date_{}_c_{}_input_sizes_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}_{}_{}_{}year_train.pth".format(
                norm, train_begin_date, c, input_sizes, prediction_size, window, hidden_dim, num_layers, model_state, freqs, run_year)
            val_model_name = model_dir + "multicross_{}norm_train_begin_date_{}_c_{}_input_sizes_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}_{}_{}_{}year_val.pth".format(
                norm, train_begin_date, c, input_sizes, prediction_size, window, hidden_dim, num_layers, model_state, freqs, run_year)
            if model_state == 'independent':
                model = MultiGRU_Independent(input_sizes, hidden_dim, num_layers, output_dim, dropout_rate, c, num_heads, device)
                model.to(device)
            else:
                model = multigru(input_sizes[0], hidden_dim, num_layers, output_dim, dropout_rate, c, num_heads, device)
                model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08)
            if os.path.exists(train_model_name):
                checkpoint = torch.load(train_model_name,map_location=device)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                logger.info('Loaded model from epoch {}'.format(start_epoch))
            else:
                start_epoch = 0
                logger.info('No saved model found, starting training from scratch')

            logger.info('test_flag: {}, train_model_name: {},  \n \
                        train_begin_date: {}, train_end_date: {}, \
                        valid_begin_date: {}, valid_end_date: {},\
                        test_begin_date: {}, test_end_date: {}, \n log_path: {}, factor_name: {}, \n \
                        threshold_of_grow_and_decline: {}, prediction_size: {}, window: {}, \
                        input_size: {}, hidden_dim: {}, num_layers: {}, output_dim: {}, \n \
                        epochs: {}, lr: {}, batch_size: {}, device: {}, c: {}, num_heads: {}, \
                        model_state: {}, freq: {},  run_year: {}'.format(
                                                                    test_flag, train_model_name,  train_begin_date, train_end_date,
                                                                    valid_begin_date, valid_end_date, test_begin_date, test_end_date,
                                                                    log_path, factor_name, threshold_of_grow_and_decline, prediction_size, window,
                                                                    input_sizes, hidden_dim, num_layers, output_dim,
                                                                    epochs, lr, batch_size, device, c, num_heads, model_state, freqs, run_year))
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total number of parameters: {total_params}")
            logger.info('Loading Dataset...')
            datasets = []
            train_datasets = []
            valid_datasets = []
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
                elif input_size == 19:
                    adds = True
                elif input_size == 20:
                    outstand = True
                    drop = True
                elif input_size == 14:
                    adds = True
                    drop = True
                elif input_size == 26:
                    outstand = True
                    ex = False
                    drop = False
                elif input_size == 44:
                    outstand = True
                    drop = True
                else:
                    pass
                freq = freqs[index]
                dataset , df_key = time_series_dataset_dict_cross(
                            begin_date = train_begin_date,  end_date = test_end_date,
                            threshold_of_grow_and_decline = threshold_of_grow_and_decline, 
                            prediction_size = prediction_size, adds = adds, window = window, 
                            drop = drop, norm = norm , freq = freq, ex = ex, outstand = outstand, drop0930 = drop0930)
                datasets.append(dataset)
                df_keys.append(df_key)
                train_dataset = {}
                valid_dataset = {}
                test_dataset = {}
                for timestamp, data in dataset.items():
                    if pd.Timestamp(test_begin_date) <= timestamp <= pd.Timestamp(test_end_date):
                        test_dataset[timestamp] = data
                    if pd.Timestamp(train_begin_date)<=timestamp< pd.Timestamp(train_end_date):
                        train_dataset[timestamp] = data
                    elif pd.Timestamp(valid_begin_date)<=timestamp< pd.Timestamp(valid_end_date):
                        valid_dataset[timestamp] = data
                train_datasets.append(train_dataset)
                valid_datasets.append(valid_dataset)        
                test_datasets.append(test_dataset)
                test_key = df_key[df_key['date']>=pd.Timestamp(test_begin_date)]
                test_key = test_key.sort_values(by = ['date','symbol'])
                test_key = test_key.reset_index(drop=True)
                test_keys.append(test_key)
            if loss =='MSE':
                criterion = torch.nn.MSELoss().to(device)
            elif loss == 'Cos':
                criterion = CosineLoss().to(device)
            elif loss =='IC':
                criterion = ICBasedLoss().to(device)
            elif loss == 'multiIC':
                criterion = multiIC(0.01).to(device)
            scaler = GradScaler()
            train_min_loss = np.Inf
            val_min_loss = np.Inf

            for epoch in range(start_epoch, epochs):
                with autocast():
                    train_loss = train_one_epoch_dict_2(model, train_datasets, optimizer, criterion, logger, device, epoch, scaler)
                    if np.mean(train_loss) < train_min_loss:
                        # 保存模型
                        train_min_loss = np.mean(train_loss)
                        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()}
                        torch.save(state, train_model_name)
                        logger.info('Epoch: {} best train model saved successfully!'.format(epoch))
                    logger.info('Epoch: {}, train loss: {:.10f}, train min loss: {:.10f}'.format(epoch, np.mean(train_loss), train_min_loss))

                    valid_loss = test_dict_2(model, valid_datasets, criterion, logger, device)
                    scheduler.step(np.mean(valid_loss))
                    if (val_min_loss==np.Inf) or (np.mean(valid_loss) < val_min_loss):
                        val_min_loss = np.mean(valid_loss)
                        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()}
                        torch.save(state, val_model_name)
                        logger.info('Epoch: {} best valid model saved successfully!'.format(epoch))
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience:
                            logger.info('Early stopping!')
                            break
                    logger.info('Epoch: {}, val loss: {:.10f}, val min loss: {:.10f}, lr: {:.10f}'.format(epoch, np.mean(valid_loss), val_min_loss, optimizer.param_groups[0]['lr']))
                if test_flag:
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
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)  

        if test_flag:
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
    main(test_flag=True, model_dir="models/", run_year=1, 
    train_begin_dates = '2015-03-03',
    train_end_dates = '2018-12-31' ,
    valid_begin_dates = '2019-01-01',
    valid_end_dates = '2019-05-31',
    test_begin_dates = '2019-06-01',
    test_end_dates = '2020-05-31')