import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from create_data import smile_to_graph


class EvaluateDataset(Dataset):
    def __init__(self, xd, xt):
        super(EvaluateDataset, self).__init__()
        
        ## AASeq tokens
        self.aaseq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        self.aaseq_dict = {v:(i+1) for i,v in enumerate(self.aaseq_voc)}
        self.aaseq_maxlen = 1000
        
        ## Process
        self.data_list = self._process(xd, xt)


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        return self.data_list[idx]
    

    def _process(self, xd, xt):
        '''
        xd : list of SMILES strings
        xt : list of AASeq strings
        '''
        assert len(xd) == len(xt), "The two lists must be the same length!"
                
        data_list = []
        for smiles, target in zip(xd, xt):
            ## Convert SMILES to molecular representation using rdkit
            smiles_graph = smile_to_graph(smiles)
            c_size, features, edge_index = smiles_graph
            features = np.asarray(features)
            edge_index = np.asarray(edge_index)
            ## Convert AASeq to index vector
            prot_vec = self._seq_cat(target)
            prot_vec = np.expand_dims(np.asarray(prot_vec), axis=0)
            ## Make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            GCNData.target = torch.LongTensor(prot_vec)
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            ## append graph, label and target sequence to data list
            data_list.append(GCNData)
        
        return data_list
        
        
    def _seq_cat(self, prot):
        x = np.zeros(self.aaseq_maxlen)
        for i, ch in enumerate(prot[:self.aaseq_maxlen]): 
            x[i] = self.aaseq_dict[ch]
        return x  


class GraphDTA:
    def __init__(self, modeltype, n_output=1, num_features_xd=78, num_features_xt=25, n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, device=None):
        ## parameters
        self.modeltype = modeltype
        self.modelparams = {
            'n_output':n_output,
            'num_features_xd':num_features_xd,
            'num_features_xt':num_features_xt,
            'n_filters':n_filters,
            'embed_dim':embed_dim,
            'output_dim':output_dim,
            'dropout':dropout,
        }
        self.device = device
        ## model init
        self.model = None
        for modeling in [GINConvNet, GATNet, GAT_GCN, GCNNet]:
            if modeltype == modeling.__name__:
                self.model = modeling(**self.modelparams).to(self.device)
        assert self.model is not None, f"[ERROR] {modeltype} is not available!!!"
        
        
    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        return self


    def evaluate(self, xd, xt, batch_size=512):
        ## Mode
        self.model.eval()
        
        ## Loader
        dataset = EvaluateDataset(xd=xd, xt=xt)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        ## Prediction
        res = torch.Tensor()
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                output = self.model(data)
                res = torch.cat((res, output.cpu()), 0)

        return res.numpy().flatten()
        


def main():
    drugs = {
        'abt737':'CN(C)CCC(CSC1=CC=CC=C1)NC2=C(C=C(C=C2)S(=O)(=O)NC(=O)C3=CC=C(C=C3)N4CCN(CC4)CC5=CC=CC=C5C6=CC=C(C=C6)Cl)[N+](=O)[O-]',
        'navitoclax':'CC1(CCC(=C(C1)CN2CCN(CC2)C3=CC=C(C=C3)C(=O)NS(=O)(=O)C4=CC(=C(C=C4)NC(CCN5CCOCC5)CSC6=CC=CC=C6)S(=O)(=O)C(F)(F)F)C7=CC=C(C=C7)Cl)C',
    }
    
    targets = {
        'bcl2':'MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPAASRDPVARTSPLQTPAAPGAAAGPALSPVPPVVHLTLRQAGDDFSRRYRRDFAEMSSQLHLTPFTARGRFATVVEELFRDGVNWGRIVAFFEFGGVMCVESVNREMSPLVDNIALWMTEYLNRHLHTWIQDNGGWDAFVELYGPSMRPLFDFSWLSLKTLLSLALVGACITLGAYLGHK',
        'bclxl':'MSQSNRELVVDFLSYKLSQKGYSWSQFSDVEENRTEAPEGTESEMETPSAINGNPSWHLADSPAVNGATGHSSSLDAREVIPMAAVKQALREAGDEFELRYRRAFSDLTSQLHITPGTAYQSFEQVVNELFRDGVNWGRIVAFFSFGGALCVESVDKEMQVLVSRIAAWMATYLNDHLEPWIQENGGWDTFVELYGNNAAAESRKGQERFNRWFLTGMTVAGVVLLGSLFSRK',
        'bclw':'MATPASAPDTRALVADFVGYKLRQKGYVCGAGPGEGPAADPLHQAMRAAGDEFETRFRRTFSDLAAQLHVTPGSAQQRFTQVSDELFQGGPNWGRLVAFFVFGAALCAESVNKEMEPLVGQVQEWMVAYLETQLADWIHSSGGWAEFTALYGDGALEEARRLREGNWASVRTVLTGAVALGALVTVGAFFASK',
    }
    
    pairs = []
    xd = []
    xt = []
    
    for kd, vd in drugs.items():
        for kt, vt in targets.items():
            pairs.append((kd,kt))
            xd.append(vd)
            xt.append(vt)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else "cpu")
    
    modeltype = 'GINConvNet'
    model = GraphDTA(modeltype, device=device).load(f'model_{modeltype}_kiba.model')
    
    print(model.evaluate(xd, xt))
    