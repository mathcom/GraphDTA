import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *


## training function at each epoch
def train(model, device, train_loader, optimizer, loss_fn, epoch, log_interval):
    print(f'Training on {len(train_loader.dataset)} samples...')
    model.train()
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        output = model(data)
        
        optimizer.zero_grad()
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'[INFO] train epoch: {epoch:06d} ({int(100*batch_idx/len(train_loader)):02d}%)\tLoss: {loss.item():.6f}')
            #print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
            #                                                               batch_idx * train_loader.batch_size,
            #                                                               len(train_loader.dataset),
            #                                                               100. * batch_idx / len(train_loader),
            #                                                               loss.item()))

def predicting(model, device, loader):
    model.eval()
    
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    
    print(f'Make prediction for {len(loader.dataset)} samples...')
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def main():
    datasets = [['davis','kiba'][int(sys.argv[1])]] 
    modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
    model_st = modeling.__name__

    cuda_name = "cuda:0"
    if len(sys.argv)>3:
        cuda_name = "cuda:" + str(int(sys.argv[3])) 
    print(f'cuda_name: {cuda_name}')

    ## Global Variables
    print(f'Learning rate: {LR}')
    print(f'Epochs: {NUM_EPOCHS}')

    ## Main program: iterate over different datasets
    for dataset in datasets:
        print()
        print(f'running on {model_st}_{dataset}')
        
        processed_data_file_train = os.path.join('data', 'processed', f'{dataset}_train.pt')
        processed_data_file_test = os.path.join('data', 'processed', f'{dataset}_test.pt')
        
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
            print('please run create_data.py to prepare data in pytorch format!')
        else:
            train_data = TestbedDataset(root='data', dataset=f'{dataset}_train')
            test_data = TestbedDataset(root='data', dataset=f'{dataset}_test')
            
            ## make data PyTorch mini-batch processing ready
            train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

            ## training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            
            best_mse = 1000
            best_ci = 0
            best_epoch = -1
            
            model_file_name = f'model_{model_st}_{dataset}.model'
            result_file_name = f'result_{model_st}_{dataset}.csv'
            
            for epoch in range(NUM_EPOCHS):
                train(model, device, train_loader, optimizer, loss_fn, epoch+1, LOG_INTERVAL)
                
                G, P = predicting(model, device, test_loader)
                
                ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
                
                if ret[1] < best_mse:
                    torch.save(model.state_dict(), model_file_name)
                    with open(result_file_name, 'w') as f:
                        f.write(','.join(map(str,ret)))
                    best_epoch = epoch+1
                    best_mse = ret[1]
                    best_ci = ret[-1]
                    print(f'rmse improved at epoch {best_epoch}; best_mse,best_ci: {best_mse} {best_ci} {model_st} {dataset}')
                else:
                    print(f'{ret[1]} No improvement since epoch {best_epoch}; best_mse,best_ci: {best_mse} {best_ci} {model_st} {dataset}')


if __name__=='__main__':
    
    TRAIN_BATCH_SIZE = 512
    TEST_BATCH_SIZE = 512
    LR = 0.0005
    LOG_INTERVAL = 20
    NUM_EPOCHS = 1000
    
    main()