import os
import torch
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset


def get_ogb_data(d_name='arxiv', print_stats=True):
    # download and load the obg dataset
    root = os.path.join(os.path.realpath('../'), 'data', d_name)
    dataset = PygNodePropPredDataset(f'ogbn-{d_name}', root)
    
    # split_idx contains a dictionary of train, validation and test node indices
    split_idx = dataset.get_idx_split()
    
    # loading the dataset
    data = dataset[0]
    num_classes = dataset.num_classes
    
    # get masks
    data.train_mask = mask_generation(split_idx['train'], data.num_nodes)
    data.valid_mask = mask_generation(split_idx['valid'], data.num_nodes)
    data.test_mask = mask_generation(split_idx['test'], data.num_nodes)
    
    data.n_id = torch.arange(data.num_nodes)
    
    # # load integer to real product category from label mapping provided inside the dataset
    # if d_name == 'arxiv':
    #     df = pd.read_csv('../data/arxiv/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz')
    # else:
    #     df = pd.read_csv('../data/products/ogbn_products/mapping/labelidx2productcategory.csv.gz')
    
    # lets check some graph statistics of ogb-product graph
    if print_stats:
        print("Number of nodes in the graph:", data.num_nodes)
        print("Number of edges in the graph:", data.num_edges)
        
        # lets check the node ids distribution of train, test and val
        print('Number of training nodes:', split_idx['train'].size(0))
        print('Number of validation nodes:', split_idx['valid'].size(0))
        print('Number of test nodes:', split_idx['test'].size(0))
        
        print("Node feature matrix with shape:", data.x.shape) # [num_nodes, num_node_features]
        print("Graph connectivity in COO format with shape:", data.edge_index.shape) # [2, num_edges]
        print("Target to train against :", data.y.shape) 
        print("Node feature length", dataset.num_features)
        print(f"number of target categories: {data.y.unique().size(0)}")
        
    return data, split_idx, num_classes
    

def mask_generation(index, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[index] = 1
    return mask