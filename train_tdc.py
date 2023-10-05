import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tdc.multi_pred import DTI

from utils import TestbedDataset
from create_data import seq_cat, smile_to_graph
from interface import GraphDTA, Trainer


def get_smile_graph(list_of_smiles):
    list_of_smiles = list(set(list_of_smiles))
    return {s:smile_to_graph(s) for s in list_of_smiles}
       

def create_tdc_dataset(df, fold, smile_graph):
    ## params
    max_seqlen = 1000
    aaseq_dict = {v:(i+1) for i,v in enumerate("ABCDEFGHIKLMNOPQRSTUVWXYZ")}
    
    ## preprocess
    xd = df.loc[:,'Drug'].values
    xt = np.asarray([seq_cat(t, max_seqlen, aaseq_dict) for t in df.loc[:,'Target'].values])
    y  = df.loc[:,'Y'].values
    
    ## dataset
    return TestbedDataset(root='data', dataset=f'tdc_{fold}', xd=xd, xt=xt, y=y, smile_graph=smile_graph, use_tqdm=True)
    

def main():
    ## TDC data download
    dataset_name = 'tdc'
    data = DTI(name = 'BindingDB_Kd')
    
    ## Data preprocessing
    data.convert_to_log(form = 'binding')
    df_whole = data.harmonize_affinities(mode = 'max_affinity')
    
    ## Data split
    frames = data.get_split(method='cold_drug', seed=2023, frac=[0.8, 0.1, 0.1])
    df_tr = frames['train']
    df_va = frames['valid']
    df_te = frames['test']
    
    ## Create a mapping table for smiles2graph
    smile_graph = get_smile_graph(df_whole.loc[:,'Drug'].values)
    
    ## PyTorch Geometric Dataset
    train_data = create_tdc_dataset(df_tr, 'train', smile_graph)
    test_data = create_tdc_dataset(df_te, 'test', smile_graph)
    valid_data = create_tdc_dataset(df_va, 'valid', smile_graph)
    
    ## Check GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else "cpu")
    
    ## Select Model
    modeling = ['GINConvNet', 'GATNet', 'GAT_GCN', 'GCNNet'][int(sys.argv[1])]
    model = GraphDTA(modeling, device=device)
    
    ## Initialize a Trainer
    trainer = Trainer(model.model, model.device)
    
    ## Start model training
    trainer.train(train_data, test_data, valid_data,
                  ckpt_dir='./ckpt', ckpt_filename=f'{modeling}_tdc')
                  
if __name__=='__main__':
    main()