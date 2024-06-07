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
from datasets.dataset import process_raw_df
from engine import *
pd.options.mode.chained_assignment = None

def main(test_flag=False, model_dir="models/"):
    train_begin_date = '2015-01-03'
    train_end_date = '2018-12-31' 
    valid_begin_date = '2019-01-03'
    valid_end_date = '2019-05-31'
    test_begin_date = '2019-06-01'
    test_end_date = '2020-06-30'
    
    log_dir = 'log/'

    input_size = 24
    output_dim = 1
    epochs = 50
    lr = 0.001
    weight_decay=1e-5
    batch_size = 256  
    threshold_of_grow_and_decline = 0.5
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    param_grid = {
        'prediction_size': [10],
        'window': [30],
        'hidden_dim': [256 , 512],
        'num_layers': [3, 5]
    }
    param_list = list(ParameterGrid(param_grid))
    # prediction_size = 3
    # window = 36
    # hidden_dim = 512
    # num_layers = 3
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    #                 filename=log_path, filemode='w')
    # logger = logging.getLogger()

    best_params = None
    best_val_loss = np.Inf

    for params in param_list:
        prediction_size = params['prediction_size']
        window = params['window']
        hidden_dim = params['hidden_dim']
        num_layers = params['num_layers']
        train_model_name = model_dir + "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}_train.pth".format(input_size , prediction_size , window , hidden_dim , num_layers)
        val_model_name = model_dir + "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}_val.pth".format(input_size , prediction_size , window , hidden_dim , num_layers)
        best_model_name = model_dir + "input_size_{}_best.pth".format(input_size)
        factor_name = "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}".format(input_size , prediction_size , window , hidden_dim , num_layers)
        # Log
        log_path = log_dir + "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}.log".format(input_size , prediction_size , window , hidden_dim , num_layers)

        model = LSTM(input_size, hidden_dim, num_layers, output_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08)

        if os.path.exists(train_model_name):
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=log_path, filemode='a')
            logger = logging.getLogger()
            checkpoint = torch.load(train_model_name,map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']

            logger.info('Loaded model from epoch {}'.format(start_epoch))
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=log_path, filemode='w')
            logger = logging.getLogger()
            start_epoch = 0
            logger.info('No saved model found, starting training from scratch')

        logger.info('test_flag: {}, train_model_name: {}, val_model_name: {}, \n \
                    best_model_name : {} \n \
                    train_begin_date: {}, train_end_date: {}, \
                    valid_begin_date: {}, valid_end_date: {},\
                    test_begin_date: {}, test_end_date: {}, \n log_path: {}, factor_name: {}, \n \
                    threshold_of_grow_and_decline: {}, prediction_size: {}, window: {}, \
                    input_size: {}, hidden_dim: {}, num_layers: {}, output_dim: {}, \n \
                    epochs: {}, lr: {}, batch_size: {}, device: {}'.format(
                                                                test_flag, train_model_name, val_model_name, best_model_name, train_begin_date, train_end_date,
                                                                valid_begin_date, valid_end_date, test_begin_date, test_end_date,
                                                                log_path, factor_name, threshold_of_grow_and_decline, prediction_size, window,
                                                                input_size, hidden_dim, num_layers, output_dim,
                                                                epochs, lr, batch_size, device))

        train_X, train_y, valid_X ,valid_y , test_X , test_y = process_raw_df(
                                                            train_begin_date = train_begin_date, train_end_date = train_end_date, 
                                                            valid_begin_date = valid_begin_date, valid_end_date = valid_end_date,
                                                            test_begin_date = test_begin_date , test_end_date = test_end_date,
                                                            threshold_of_grow_and_decline = threshold_of_grow_and_decline, 
                                                            prediction_size = prediction_size, window = window)

        train_dataset = TensorDataset(train_X, train_y)
        valid_dataset = TensorDataset(valid_X, valid_y)
        test_dataset = TensorDataset(test_X, test_y)
        shuffle_dataset = True  

        criterion = torch.nn.MSELoss(reduction='mean').to(device)
        train_min_loss = np.Inf
        val_min_loss = np.Inf
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total number of parameters: {total_params}")

        for epoch in range(start_epoch, epochs):
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_dataset, pin_memory=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle_dataset, pin_memory=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_dataset, pin_memory=True)

            train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, logger, device, epoch)
            
            if np.mean(train_loss) < train_min_loss:
                # 保存模型
                train_min_loss = np.mean(train_loss)
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()}
                torch.save(state, train_model_name)
                logger.info('Epoch: {} best train model saved successfully!'.format(epoch))
            logger.info('Epoch: {}, train loss: {:.10f}, train min loss: {:.10f}'.format(epoch, np.mean(train_loss), train_min_loss))

            valid_loss = test(model, valid_dataloader, criterion, logger, device)
            scheduler.step(np.mean(valid_loss))

            if np.mean(valid_loss) < val_min_loss:
                best_epoch = epoch
                val_min_loss = np.mean(valid_loss)
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()}
                val_model_epoch = model_dir + "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}_epoch_{}_val.pth".format(input_size , prediction_size , window , hidden_dim , num_layers, epoch)
                torch.save(state, val_model_epoch)
                logger.info('Epoch: {} best valid model saved successfully!'.format(epoch))
            logger.info('Epoch: {}, val loss: {:.10f}, val min loss: {:.10f}, best epoch: {}'.format(epoch, np.mean(valid_loss), val_min_loss, best_epoch))

        if val_min_loss < best_val_loss:
            best_params = params
            best_val_loss = val_min_loss
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()}
            torch.save(state, best_model_name)
            logger.info('val_min_loss: {:.10f}, best_val_loss: {:.10f}'.format(val_min_loss, best_val_loss))

        logger.info("best hyper params : prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}".format(
                best_params['prediction_size'] , best_params['window'] , best_params['hidden_dim'] , num_layers = best_params['num_layers']))
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        if test_flag:
            logger.info('Testing mode')
            model.eval()

            with torch.no_grad():
                outputs = []
                for batch_idx, (data, target) in enumerate(test_dataloader):
                    data, target = data.to(device), target.to(device)
                    outputs.append(model(data))

            test_dataloader[factor_name] = torch.cat(outputs).cpu().numpy()
            test_dataloader.to_parquet('factor/{}.parq'.format(factor_name))

if __name__ == "__main__":
    main(test_flag=False, model_dir="models/")