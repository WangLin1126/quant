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
    valid_end_date = '2019-05-31'
    test_begin_date = '2019-06-01'
    test_end_date = '2020-06-30'
    log_dir = 'log/'
    loss = 'MSE'
    norm = True
    input_size = 20
    output_dim = 1
    epochs = 100
    lr = 0.001
    weight_decay=1e-5
    batch_size = 256  
    threshold_of_grow_and_decline = 0.5
    patience = 15
    fold = 10
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    param_grid = {
        'prediction_size': [10],
        'window': [30],
        'hidden_dim': [256],
        'num_layers': [3]
    }
    param_list = list(ParameterGrid(param_grid))

    train_begin = pd.Timestamp(train_begin_date)
    valid_end = pd.Timestamp(valid_end_date)
    total_days = (valid_end - train_begin).days
    days_per_fold = total_days // fold
    validation_dates = []

    for i in range(fold):
        validation_start = train_begin + pd.Timedelta(days=i*days_per_fold)
        validation_end = validation_start + pd.Timedelta(days=days_per_fold)
        validation_dates.append((validation_start, validation_end))

    for params in param_list:
        prediction_size = params['prediction_size']
        window = params['window']
        hidden_dim = params['hidden_dim']
        num_layers = params['num_layers']
        factor_name = "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}".format(input_size , prediction_size , window , hidden_dim , num_layers)
        log_path = log_dir + "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}.log".format(input_size , prediction_size , window , hidden_dim , num_layers)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                                filename=log_path, filemode='a')
        logger = logging.getLogger()

        logger.info('Loading Dataset...')
        if input_size == 6:
            adds = False
        else:
            adds = True
        dataset , df_keys = time_series_dataset_dict_cross(
                    train_begin_date = train_begin_date,  test_end_date = test_end_date,
                    threshold_of_grow_and_decline = threshold_of_grow_and_decline, 
                    prediction_size = prediction_size, adds = adds,
                    window = window, norm = norm)
        
        test_keys = df_keys[df_keys['date']>=pd.Timestamp(test_begin_date)]
        test_keys = test_keys.sort_values(by = ['date','symbol'])
        test_keys = test_keys.reset_index(drop=True)

        for i in range(fold):    
            validation_start, validation_end = validation_dates[i]
            train_dataset = {}
            val_dataset = {}
            test_dataset = {}
            for timestamp, data in dataset.items():
                if pd.Timestamp(train_begin_date) <= timestamp < validation_start:
                    train_dataset[timestamp] = data
                elif validation_start <= timestamp < validation_end:
                    val_dataset[timestamp] = data
                elif validation_end <= timestamp < pd.Timestamp(test_begin_date):
                    train_dataset[timestamp] = data
                elif pd.Timestamp(test_begin_date) <= timestamp <= pd.Timestamp(test_end_date):
                    test_dataset[timestamp] = data
    
            train_model_name = model_dir + "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}_fold_{}_train.pth".format(input_size , prediction_size , window , hidden_dim , num_layers, i)
            val_model_name = model_dir + "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}_fold_{}_val.pth".format(input_size , prediction_size , window , hidden_dim , num_layers, i)
            model = LSTM(input_size, hidden_dim, num_layers, output_dim).to(device)
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

            logger.info('test_flag: {}, train_model_name: {}, \n \
                        train_begin_date: {}, valid_end_date: {},\
                        test_begin_date: {}, test_end_date: {}, \n \
                        log_path: {}, factor_name: {}, \n \
                        threshold_of_grow_and_decline: {}, prediction_size: {}, window: {}, fold: {}\
                        input_size: {}, hidden_dim: {}, num_layers: {}, output_dim: {}, \n \
                        epochs: {}, lr: {}, batch_size: {}, device: {}'.format(
                                                                    test_flag, train_model_name,  train_begin_date, valid_end_date, test_begin_date, test_end_date,
                                                                    log_path, factor_name, threshold_of_grow_and_decline, prediction_size, window, i,
                                                                    input_size, hidden_dim, num_layers, output_dim,
                                                                    epochs, lr, batch_size, device))
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total number of parameters: {total_params}")

            if loss =='MSE':
                criterion = torch.nn.MSELoss().to(device)
            elif loss == 'Cos':
                criterion = CosineLoss().to(device)
            elif loss =='IC':
                criterion = ICBasedLoss().to(device)
            train_min_loss = np.Inf
            val_min_loss = np.Inf

            for epoch in range(start_epoch, epochs):
                train_loss = train_one_epoch_dict(model, train_dataset, optimizer, criterion, logger, device, epoch)
                valid_loss = test_dict(model, val_dataset, criterion, logger, device)

                if np.mean(train_loss) < train_min_loss:
                    # 保存模型
                    train_min_loss = np.mean(train_loss)
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()}
                    torch.save(state, train_model_name)
                    logger.info('Epoch: {} best train model saved successfully!'.format(epoch))
                logger.info('Epoch: {}, fold: {}, train loss: {:.10f}, train min loss: {:.10f}'.format(epoch, i, np.mean(train_loss), train_min_loss))
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
            print ("backtest data generate begin: {}".format(factor_name))
            backtest = None
            for i in range(fold):
                model_name = './models/input_size_{}_prediction_size_10_window_30_hidden_dim_256_num_layers_3_fold_{}_val.pth'.format(input_size , i)
                model = LSTM(input_size, hidden_dim, num_layers, output_dim).to(device)
                checkpoint = torch.load(model_name, map_location = device)
                model.load_state_dict(checkpoint['model'])
                model.eval()
                with torch.no_grad():
                    outputs = []
                    keys = list(test_dataset.keys())
                    for key in keys:
                        data, target = test_dataset[key].tensors[0] , test_dataset[key].tensors[1]
                        data, target = data.to(device), target.to(device)
                        target = target.view(-1)
                        outputs.append(model(data))
                    if backtest is None:
                        backtest = torch.cat(outputs).cpu().numpy()
                    else:
                        backtest += torch.cat(outputs).cpu().numpy()
                        
            test_keys[factor_name] = backtest
            test_keys.to_parquet('factor/{}.parq'.format(factor_name))    
            factor_dir = os.path.join(os.path.expanduser('~'), 'backtest/factors', factor_name, '1000')
            if not os.path.exists(factor_dir):
                os.makedirs(factor_dir)

            data_file = os.path.join(os.getcwd(), 'factor', f'{factor_name}.parq')
            destination_file = os.path.join(factor_dir, 'data.parq')
            if os.path.isfile(data_file):
                shutil.copyfile(data_file, destination_file)
                print ("backtest data generate end: {}".format(factor_name))
            else:
                print(f"data {data_file} does not exist.")                
        for handler in logger.handlers[:]:
            logger.removeHandler(handler) 

if __name__ == "__main__":
    main(test_flag=True, model_dir="models/")