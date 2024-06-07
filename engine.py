import torch
import numpy as np
import random
import sys
def train_one_epoch(model, dataloader, optimizer, criterion, logger, device, epoch):
    random.seed(epoch)
    model.train()
    train_loss = []
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_loss.append(loss.item())
        if batch_idx % 1000 == 0:
            logger.info('Epoch: {}, Train Loss: {:.10f}, Batch Index: [{}]/[{}]'.format(epoch, loss.item(), batch_idx, len(dataloader)))
        loss.backward()
        optimizer.step()
    return train_loss

def train_one_epoch_dict(model, dict, optimizer, criterion, logger, device, epoch, scaler = None):
    random.seed(epoch)
    model.train()
    train_loss = []
    keys = list(dict.keys())
    random.shuffle(keys)
    i = 0
    for key in keys:
        data, target = dict[key].tensors[0] , dict[key].tensors[1]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data).view(-1)
        target = target.view(-1)
        loss = criterion(outputs, target)
        train_loss.append(loss.item())
        if i % 100 == 0:
            logger.info('Epoch: {}, Train Loss: {:.10f}, Batch Index: [{}]/[{}]'.format(epoch, loss.item(), i, len(dict)))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        i+=1
    return np.mean(train_loss)

def train_one_epoch_dict_2(model, dicts, optimizer, criterion, logger, device, epoch, scaler = None):
    random.seed(epoch)
    model.train()
    train_loss = []
    keys = list(dicts[0].keys())
    random.shuffle(keys)
    i = 0
    for key in keys:
        datas = []
        targets = []
        for index in range(len(dicts)):
            data, target = dicts[index][key].tensors[0] , dicts[index][key].tensors[1]
            data, target = data.to(device), target.to(device)
            datas.append(data)
            targets.append(target)
        optimizer.zero_grad()
        output = model(datas)
        if torch.isnan(output).any():
            print("output tensor contains NaN values.")
        else:
            pass
        output = output.view(-1)
        loss = criterion(output, targets[0].view(-1))
        train_loss.append(loss.item())
        if i % 100 == 0:
            logger.info('Epoch: {}, Train Loss: {:.10f}, Batch Index: [{}]/[{}]'.format(epoch, loss.item(), i, len(dicts[0])))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        i+=1
    return np.mean(train_loss)


def test(model, dataloader, criterion, logger, device):
    model.eval()
    test_loss = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss.append(loss.item())
            if batch_idx % 100 == 0:
                logger.info('Val Loss: {:.10f}, Batch Index: [{}]/[{}]'.format(loss.item(), batch_idx, len(dataloader)))
    return np.mean(test_loss)

def test_dict(model, dict,  criterion, logger, device):
    model.eval()
    test_loss = []
    with torch.no_grad():
        keys = list(dict.keys())
        # random.shuffle(keys)
        i = 0
        for key in keys:
            data, target = dict[key].tensors[0] , dict[key].tensors[1]
            data, target = data.to(device), target.to(device)
            outputs = model(data).view(-1)
            target = target.view(-1)
            loss = criterion(outputs, target)
            test_loss.append(loss.item())
            if i % 50 == 0:
                logger.info('Val Loss: {:.10f}, Batch Index: [{}]/[{}]'.format(loss.item(), i, len(dict)))
            i = i + 1 
    return np.mean(test_loss)

def test_dict_2(model, dicts,  criterion, logger, device):
    model.eval()
    test_loss = []
    with torch.no_grad():
        keys = list(dicts[0].keys())
        # random.shuffle(keys)
        i = 0
        for key in keys:
            datas = []
            targets = []
            for index in range(len(dicts)):
                data, target = dicts[index][key].tensors[0] , dicts[index][key].tensors[1]
                data, target = data.to(device), target.to(device)
                datas.append(data)
                targets.append(target)
            output = model(datas)
            output = output.view(-1)
            loss = criterion(output, targets[0])
            test_loss.append(loss.item())
            if i % 50 == 0:
                logger.info('Val Loss: {:.10f}, Batch Index: [{}]/[{}]'.format(loss.item(), i, len(dicts[0])))
            i = i + 1 
    return np.mean(test_loss)