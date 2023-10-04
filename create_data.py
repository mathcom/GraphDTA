import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index


def seq_cat(prot, max_seq_len, seq_dict):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  


def main():
    # from DeepDTA data
    all_prots = []
    DATASETS = ['kiba', 'davis']
    
    for dataset in DATASETS:
        print(f'[INFO] Convert data from DeepDTA for {dataset}')
        
        ## Directory & Filepathes
        fpath = os.path.join('data', dataset)
        train_fold = json.load(open(os.path.join(fpath, 'folds', 'train_fold_setting1.txt')))
        train_fold = [ee for e in train_fold for ee in e ]
        valid_fold = json.load(open(os.path.join(fpath, 'folds', 'test_fold_setting1.txt')))
        ligands = json.load(open(os.path.join(fpath, "ligands_can.txt")), object_pairs_hook=OrderedDict)
        proteins = json.load(open(os.path.join(fpath, "proteins.txt")), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(os.path.join(fpath, "Y"),"rb"), encoding='latin1')
        
        ## Data loading
        print(f'[DEBUG] Data Loading')
        drugs = []
        prots = []
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
            drugs.append(lg)
            
        for t in proteins.keys():
            prots.append(proteins[t])
        
        ## Normalization
        if dataset == 'davis':
            affinity = [-np.log10(y * 1e-9) for y in affinity]
        
        affinity = np.asarray(affinity)
        
        ## Fold
        print(f'[DEBUG] Partioning into training and test folds')
        opts = ['train','test']
        for opt in opts:
            rows, cols = np.where(np.isnan(affinity)==False)  
            
            if opt =='train':
                rows,cols = rows[train_fold], cols[train_fold]
            elif opt =='test':
                rows,cols = rows[valid_fold], cols[valid_fold]
            
            with open(os.path.join('data', f'{dataset}_{opt}.csv'), 'w') as f:
                f.write('compound_iso_smiles,target_sequence,affinity\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [ drugs[rows[pair_ind]]  ]
                    ls += [ prots[cols[pair_ind]]  ]
                    ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                    f.write(','.join(map(str,ls)) + '\n')       
        
        print()
        print(f'[INFO] dataset: {dataset}')
        print(f'[INFO] train_fold: {len(train_fold)}')
        print(f'[INFO] test_fold: {len(valid_fold)}')
        print(f'[INFO] len(set(drugs)), len(set(prots)): {len(set(drugs))}, {len(set(prots))}')
        
        all_prots += list(set(prots))
        
        
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
    max_seq_len = 1000

    compound_iso_smiles = []
    for dt_name in DATASETS:
        opts = ['train', 'test']
        for opt in opts:
            df = pd.read_csv(os.path.join('data', f'{dt_name}_{opt}.csv'))
            compound_iso_smiles += list( df['compound_iso_smiles'] )
            
    compound_iso_smiles = set(compound_iso_smiles)
    
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g


    ## convert to PyTorch data format
    for dataset in DATASETS:
        processed_data_file_train = os.path.join('data', 'processed', f'{dataset}_train.pt')
        processed_data_file_test = os.path.join('data', 'processed', f'{dataset}_test.pt')
        
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
            
            ## training data
            df = pd.read_csv(os.path.join('data', f'{dataset}_train.csv'))
            train_Y     = df['affinity'].values
            train_drugs = df['compound_iso_smiles'].values
            train_prots = df['target_sequence'].values.tolist()
            train_prots = np.asarray([seq_cat(t, max_seq_len, seq_dict) for t in train_prots])
            
            ## test data
            df = pd.read_csv(os.path.join('data', f'{dataset}_test.csv'))
            test_Y     = df['affinity'].values
            test_drugs = df['compound_iso_smiles'].values
            test_prots = df['target_sequence'].values.tolist()
            test_prots = np.asarray([seq_cat(t, max_seq_len, seq_dict) for t in test_prots])
            
            ## make data PyTorch Geometric ready
            print(f'[INFO] preparing {dataset}_train.pt in pytorch format!')
            train_data = TestbedDataset(root='data', dataset=f'{dataset}_train', xd=train_drugs, xt=train_prots, y=train_Y, smile_graph=smile_graph)
            
            print(f'[INFO] preparing {dataset}_test.pt in pytorch format!')
            test_data = TestbedDataset(root='data', dataset=f'{dataset}_test', xd=test_drugs, xt=test_prots, y=test_Y, smile_graph=smile_graph)
            
            print(f'{processed_data_file_train} and {processed_data_file_test} have been created')
        else:
            print(f'{processed_data_file_train} and {processed_data_file_test} are already created')        


if __name__=='__main__':
    main()