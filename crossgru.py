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
from datasets.dataset import *
from engine import *
from loss import *
import re
import shutil
from torch.cuda.amp import GradScaler, autocast
pd.options.mode.chained_assignment = None

def main(test_flag=False, model_dir="models/", run_year=4, 
    train_begin_dates = '2015-01-03',
    train_end_dates = '2018-12-31' ,
    valid_begin_dates = '2019-01-03',
    valid_end_dates = '2019-12-31',
    test_begin_dates = '2020-01-01',
    test_end_dates = '2020-12-31'):
    loss = 'IC'
    log_dir = 'log/'
    patience = 10
    output_dim = 1
    epochs = 100
    lr = 0.0005
    dropout_rate = 0.4
    weight_decay=1e-6
    batch_size = 256  
    num_heads=2
    norm = True
    random_sample = True
    winsor = True #去极值
    ratio = True #买卖单转化为比例
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    param_grid = {
            'prediction_size': [10],
            'window': [30],
            'hidden_dim': [64],
            'num_heads':[2],
            'num_layers': [3],
            'c': [30],
            'input_size': [22],
            'freq': ['1d'],
            'run_year': [2]
        }
    param_list = list(ParameterGrid(param_grid))
    
    for params in param_list:
        drop = False
        adds = False
        ex = False
        drop0930 = False
        outstand = False
        backtest = pd.DataFrame()
        prediction_size = params['prediction_size']
        window = params['window']
        hidden_dim = params['hidden_dim']
        num_heads = params['num_heads']
        num_layers = params['num_layers']
        c = params['c']
        input_size = params['input_size']
        if input_size == 54:
            adds = True
            ex = True
        elif input_size == 22:
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
        else:
            pass
        freq = params['freq']
        run_year = params['run_year']
        factor_name = "cross_{}winsor_{}ex_c_{}_input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_ratio_{}_{}_{}_{}year".format(
            winsor, ex, c, input_size , prediction_size , window , hidden_dim , ratio , loss, freq, run_year)
        # Log
        log_path = log_dir + "cross_{}winsor_{}ex_c_{}_input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_ratio_{}_{}_{}_{}year.log".format(
            winsor, ex, c, input_size , prediction_size , window , hidden_dim , ratio , loss , freq , run_year)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                filename=log_path, filemode='a')
        logger = logging.getLogger()
        for year in range(run_year):
            train_begin_date , train_end_date , valid_begin_date , valid_end_date , test_begin_date , test_end_date = fds.shift_days(
                [train_begin_dates, train_end_dates, valid_begin_dates, valid_end_dates, test_begin_dates, test_end_dates], year*365)
            train_model_name = model_dir + "cross_{}winsor_{}ex_train_begin_date_{}_c_{}_input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_ratio_{}_{}_{}_{}year_train.pth".format(
                winsor, ex, train_begin_date, c, input_size , prediction_size , window , hidden_dim , ratio , loss,freq, run_year)
            val_model_name = model_dir + "cross_{}winsor_{}ex_train_begin_date_{}_c_{}_input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_ratio_{}_{}_{}_{}year_val.pth".format(
                winsor, ex, train_begin_date, c, input_size , prediction_size , window , hidden_dim , ratio , loss,freq, run_year)
            
            model = CrossGRU(input_size, hidden_dim, num_layers, output_dim, dropout_rate, c, num_heads, device)
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
                        prediction_size: {}, window: {}, ration: {}, \
                        input_size: {}, hidden_dim: {}, num_layers: {}, output_dim: {}, \n \
                        epochs: {}, lr: {}, batch_size: {}, device: {}, c: {}, num_heads: {}'.format(
                                                                    test_flag, train_model_name,  train_begin_date, train_end_date,
                                                                    valid_begin_date, valid_end_date, test_begin_date, test_end_date,
                                                                    log_path, factor_name,  prediction_size, window, ratio,
                                                                    input_size, hidden_dim, num_layers, output_dim,
                                                                    epochs, lr, batch_size, device, c, num_heads))
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total number of parameters: {total_params}")
            logger.info('Loading Dataset...')
            dataset , df_keys = time_series_dataset_dict(
                        begin_date = train_begin_date,  end_date = test_end_date,
                        prediction_size = prediction_size, adds = adds, window = window, 
                        drop = drop, norm = norm , freq = freq, ex = ex, outstand = outstand, 
                        drop0930 = drop0930, ratio = ratio, winsor = winsor)
            train_dataset = {}
            valid_dataset = {}
            test_dataset = {}
            if random_sample:
                tradingdays = fds.range_trading_days(pd.Timestamp(train_begin_date), pd.Timestamp(valid_end_date), dtype=None)
                valdays = select_random_percentage(tradingdays, 0.08)
                for timestamp, data in dataset.items():
                    if pd.Timestamp(test_begin_date) <= timestamp <= pd.Timestamp(test_end_date):
                        test_dataset[timestamp] = data
                    elif timestamp in valdays:
                        valid_dataset[timestamp] = data
                    elif pd.Timestamp(train_begin_date) <= timestamp < pd.Timestamp(train_end_date):
                        train_dataset[timestamp] = data
            else:
                for timestamp, data in dataset.items():
                    if pd.Timestamp(test_begin_date) <= timestamp <= pd.Timestamp(test_end_date):
                        test_dataset[timestamp] = data
                    if pd.Timestamp(train_begin_date) <=timestamp< pd.Timestamp(train_end_date):
                        train_dataset[timestamp] = data
                    elif pd.Timestamp(valid_begin_date) <=timestamp< pd.Timestamp(valid_end_date):
                        valid_dataset[timestamp] = data
            test_keys = df_keys[df_keys['date']>=pd.Timestamp(test_begin_date)]
            test_keys = test_keys.sort_values(by = ['date','symbol'])
            test_keys = test_keys.reset_index(drop=True)
            if loss =='MSE':
                criterion = torch.nn.MSELoss().to(device)
            elif loss == 'Cos':
                criterion = CosineLoss().to(device)
            elif loss =='IC':
                criterion = ICBasedLoss().to(device)
            scaler = GradScaler()
            train_min_loss = np.Inf
            val_min_loss = np.Inf

            for epoch in range(start_epoch, epochs):
                with autocast():
                    train_loss = train_one_epoch_dict(model, train_dataset, optimizer, criterion, logger, device, epoch, scaler)
                    if np.mean(train_loss) < train_min_loss:
                        # 保存模型
                        train_min_loss = np.mean(train_loss)
                        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()}
                        torch.save(state, train_model_name)
                        logger.info('Epoch: {} best train model saved successfully!'.format(epoch))
                    logger.info('Epoch: {}, train loss: {:.10f}, train min loss: {:.10f}'.format(epoch, np.mean(train_loss), train_min_loss))

                    valid_loss = test_dict(model, valid_dataset, criterion, logger, device)
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
                        keys = list(test_dataset.keys())
                        for key in keys:
                            data, target = test_dataset[key].tensors[0] , test_dataset[key].tensors[1]
                            data, target = data.to(device), target.to(device)
                            target = target.view(-1)
                            outputs.append(model(data))
                        test_keys[factor_name] = torch.cat(outputs).cpu().numpy()
                        backtest = pd.concat([backtest, test_keys], ignore_index=True)
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
    main(test_flag=True, model_dir="models/", run_year=2, 
    train_begin_dates = '2015-01-03',
    train_end_dates = '2018-12-31' ,
    valid_begin_dates = '2019-01-01',
    valid_end_dates = '2019-05-31',
    test_begin_dates = '2019-06-01',
    test_end_dates = '2020-05-31')