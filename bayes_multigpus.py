#torchrun --nproc_per_node=2 --nnodes=1 bayes_multigpus.py

import torch
import pandas as pd
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset , TensorDataset
from sklearn.model_selection import ParameterGrid
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from model import *
from utilities import *
import os
import logging
from datasets.dataset import process_raw_df
from engine import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
pd.options.mode.chained_assignment = None
assert torch.cuda.is_available()

def main(test_flag=False, model_dir="models/"):
    dist.init_process_group("nccl")
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

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

    param_space = [
            Integer(5, 21, name='prediction_size'),
            Integer(20, 60, name='window'),
            Integer(384, 640, name='hidden_dim'),
            Integer(3, 6, name='num_layers')
    ]

    @use_named_args(param_space)
    def objective(prediction_size, window, hidden_dim, num_layers):
        prediction_size = int(prediction_size)
        window = int(window)
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)

        train_model_name = model_dir + "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}_train.pth".format(input_size , prediction_size , window , hidden_dim , num_layers)
        val_model_name = model_dir + "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}_val.pth".format(input_size , prediction_size , window , hidden_dim , num_layers)
        factor_name = "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}".format(input_size , prediction_size , window , hidden_dim , num_layers)
        # Log
        log_path = log_dir + "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}.log".format(input_size , prediction_size , window , hidden_dim , num_layers)

        model = LSTM(input_size, hidden_dim, num_layers, output_dim).to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08)

        if os.path.exists(train_model_name):
            logger = setup_logging(log_path,'a')
            checkpoint = torch.load(train_model_name,map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']

            logger.info('Loaded model from epoch {}'.format(start_epoch))
        else:
            logger = setup_logging(log_path,'w')
            start_epoch = 0
            logger.info('No saved model found, starting training from scratch')

        logger.info('test_flag: {}, train_model_name: {}, val_model_name: {}, \n \
                    train_begin_date: {}, train_end_date: {}, \
                    valid_begin_date: {}, valid_end_date: {},\
                    test_begin_date: {}, test_end_date: {}, \n log_path: {}, factor_name: {}, \n \
                    threshold_of_grow_and_decline: {}, prediction_size: {}, window: {}, \
                    input_size: {}, hidden_dim: {}, num_layers: {}, output_dim: {}, \n \
                    epochs: {}, lr: {}, batch_size: {}, device: {}'.format(
                                                                test_flag, train_model_name, val_model_name, train_begin_date, train_end_date,
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
        shuffle_dataset = False

        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_dataset, pin_memory=True, sampler=train_sampler)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle_dataset, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_dataset, pin_memory=True)

        criterion = torch.nn.MSELoss(reduction='mean').to(device)
        train_min_loss = np.Inf
        val_min_loss = np.Inf
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total number of parameters: {total_params}")

        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, logger, device, epoch)
            
            if np.mean(train_loss) < train_min_loss:
                train_min_loss = np.mean(train_loss)
                if dist.get_rank() == 0:
                    state = {'model': model.module.state_dict() if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.state_dict(), 
                             'optimizer': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()}
                    torch.save(state, train_model_name)
                    logger.info('Epoch: {} best train model saved successfully!'.format(epoch))
            logger.info('Epoch: {}, train loss: {:.10f}, train min loss: {:.10f}'.format(epoch, np.mean(train_loss), train_min_loss))

            valid_loss = test(model, valid_dataloader, criterion, logger, device)
            scheduler.step(np.mean(valid_loss))

            if np.mean(valid_loss) < val_min_loss:
                best_epoch = epoch
                val_min_loss = np.mean(valid_loss)
                val_model_epoch = model_dir + "input_size_{}_prediction_size_{}_window_{}_hidden_dim_{}_num_layers_{}_epoch_{}_val.pth".format(input_size , prediction_size , window , hidden_dim , num_layers, epoch)
                if dist.get_rank() == 0:
                    state = {'model': model.module.state_dict() if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.state_dict(), 
                             'optimizer': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()}
                    torch.save(state, val_model_epoch)
                    logger.info('Epoch: {} best valid model saved successfully!'.format(epoch))
            logger.info('Epoch: {}, val loss: {:.10f}, val min loss: {:.10f}, best epoch: {}'.format(epoch, np.mean(valid_loss), val_min_loss, best_epoch))

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
        
        return val_min_loss
    
    result = gp_minimize(objective, param_space, n_calls=30, random_state=0)

    # Print the best found parameters and the corresponding validation loss
    print("Best parameters: {}".format(result.x))
    print("Best validation loss: {:.5f}".format(result.fun))
    dist.destroy_process_group()

if __name__ == "__main__":
    main(test_flag=False, model_dir="models/")