import argparse
import numpy as np
import torch
import query_strategies
from utils import get_dataset, get_net, get_strategy
from pprint import pprint

from xyh import num_class_Tensor, writeto_file,get_new_query,calc_max_dist
import wandb


#wandb.init(project='deep-active learning')
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=10000, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="ImageNet_LT", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10","ImageNet_LT"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="RandomSampling", 
                    choices=["RandomSampling", 
                             "LeastConfidence", 
                             "MarginSampling", 
                             "EntropySampling", 
                             "LeastConfidenceDropout", 
                             "MarginSamplingDropout", 
                             "EntropySamplingDropout", 
                             "KMeansSampling",
                             "KCenterGreedy", 
                             "BALDDropout", 
                             "AdversarialBIM", 
                             "AdversarialDeepFool"], help="query strategy")
args = parser.parse_args()
pprint(vars(args))
print()

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
print("cuda is")
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

dataset = get_dataset(args.dataset_name)                   # load dataset
net = get_net(args.dataset_name, device)                   # load network
strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy
#wandb.watch(net)
# start experiment
dataset.initialize_labels(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

# round 0 accuracy
print("Round 0")
strategy.train()
preds = strategy.predict(dataset.get_test_data())
print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")

for rd in range(1, args.n_round+1):
    print(f"Round {rd}")

    # query
    query_idxs = strategy.query(args.n_query)
    #new_query_y=strategy.get_new_queryed_y(query_idxs)
    #content= num_class_Tensor(new_query_y)
    #f=open('num_class.txt','a')
    #f.writelines('new query dict:')
    #content=num_class_Tensor(get_new_query(strategy,query_idxs))
    #writeto_file('new dict ')
    #writeto_file(str(content))
    #writeto_file('new max dist ')
    #writeto_file(str(calc_max_dist(content)))
    #varience=torch.var(new_query_y.float())
    #f.writelines('new variences:')
    #writeto_file(str(varience))
    #f.close()
    # update labels
    strategy.update(query_idxs)
  #  print(query_idxs)
    strategy.train()

    # calculate accuracy
    preds = strategy.predict(dataset.get_test_data())
    print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")
    #wandb.log({'rd':rd,'accuracy':dataset.cal_test_acc(preds)})
