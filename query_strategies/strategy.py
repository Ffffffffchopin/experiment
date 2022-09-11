import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from xyh import calc_max_dist, num_class_Tensor
from xyh import writeto_file

class Strategy:
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net

    def query(self, n):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def get_new_queryed_y(self,pos_idx):
        #self.dataset.init_pool=np.zeros(self.dataset.n_pool,dtype=bool)
        #self.dataset.init_pool[pos_idx]=True
        queryed_idxs = np.arange(self.dataset.n_pool)[pos_idx]
        label_data=self.dataset.handler(self.dataset.X_train[queryed_idxs],self.dataset.Y_train[queryed_idxs])
        return label_data.Y

    def train(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        #num_class_dict= num_class_Tensor(labeled_data.Y)
        #num_class_dict=str(num_class_dict)
        #writeto_file('total dict: ')
        #writeto_file(str(num_class_dict))
        #writeto_file('max dist total: ')
        #writeto_file(str(calc_max_dist(num_class_dict)))
        
       
        
       
        
       
       # f.close()

        self.net.train(labeled_data)

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings

