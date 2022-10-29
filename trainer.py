import os
import random
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch_geometric.loader import NeighborLoader

from sage_model import SAGE
from predict import inference

from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.data = OrderedDict()
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, data, prefix='train',step=None,verbose=False):
        msg=f'{prefix} '
        for k, v in data.items():
            msg += f'{k} {v:.5f} '
            self.writer.add_scalar(f'{prefix}/{k}',v,step)

        if verbose:
            print(msg)
        with open(os.path.join(self.log_dir,f'{prefix}.txt'), 'a') as f:
            f.write(msg + '\n')



class Trainer:
    default_cfg={
        "optimizer_type": 'adam',
        "lr_cfg":{
            "lr": 0.01,
        },
        "train_log_step": 10,
        "save_interval": 50
    }

    def _init_network(self, data, num_classes, max_depth):
        # network
        self.model = SAGE(data.x.shape[1], data.x.shape[1], num_classes, n_layers=max_depth)

        # loss 
        self.val_losses = []
        self.train_losses = []
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_metrics = []

        # metrics


        # optimizer
        if self.cfg['optimizer_type']=='adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **self.cfg["lr_cfg"])
        elif self.cfg['optimizer_type']=='sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), **self.cfg["lr_cfg"])
        else:
            raise NotImplementedError
        
    def _init_loader(self, data, train_idx, test_idx, max_depth, num_neighbors=15):
        
        self.train_loader = NeighborLoader(data, input_nodes=train_idx,
                              shuffle=True, batch_size=128,
                              num_neighbors=[num_neighbors] * max_depth)

        self.test_loader = NeighborLoader(data, input_nodes=test_idx, num_neighbors=[-1],
                                    batch_size=128, shuffle=True)

        self.total_loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1],
                                    batch_size=4096, shuffle=False)
        
        

    def __init__(self, cfg=None):
        if cfg:
            self.cfg = {**self.default_cfg, **cfg}
        else:
            self.cfg = {**self.default_cfg}
        self.model_dir= 'model'
        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        self.pth_fn = os.path.join(self.model_dir,'model.pth')
        self.best_pth_fn = os.path.join(self.model_dir,'model_best.pth')
        
        

    def run(self, data, split_idx, num_classes, evaluator, depths_list, epochs):
        
        train_idx = split_idx['train']
        test_idx = split_idx['test']
        
        self._init_loader(data, train_idx, test_idx, max(depths_list))
        self._init_network(data, num_classes, max(depths_list))
        self._init_logger()
        self.evaluator = evaluator
        
        best_val_acc, step = self._load_model()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        

        for epoch in range(step, epochs):
            self.model.train()

            total_loss = 0
            val_loss = 0
            val_acc = 0
            
            print(f'\n Epoch {epoch} training:')
            
            for batch in tqdm(self.train_loader):
                
                depth = random.choice(depths_list)
                batch_size = batch.batch_size
                optimizer.zero_grad()

                out = self.model(batch.x.to(device), batch.edge_index.to(device), depth)
                out = out[:batch_size]

                batch_y = batch.y[:batch_size].to(device)
                batch_y = torch.reshape(batch_y, (-1,))

                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += float(loss)


            loss = total_loss / len(self.train_loader)
            train_acc, val_acc, test_acc = self._evaluate(data, split_idx, depths_list, device)
            
            # data logger and model saving
            log_info = {"train_loss": loss, "train_acc": train_acc, "val_acc":val_acc, "test_acc":test_acc}
            if ((epoch) % self.cfg['train_log_step']) == 0:
                self._log_data(log_info, epoch, 'train')
            
            if val_acc > best_val_acc:
                print(f'New best model validation accuracy: {val_acc:.5f}. Previous: {best_val_acc:.5f}')
                best_val_acc = val_acc
                self._save_model(epoch, best_val_acc, self.best_pth_fn)
            
            if (epoch) % self.cfg['save_interval'] == 0:
                self._save_model(epoch, best_val_acc)
            

            print(f'Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}')
            print(f'Train loss: {loss:.4f}')


    def _evaluate(self, data, split_idx, depths_list, device):
        out, _, _ = inference(self.model, self.total_loader, depths_list, device)
        
        y_true = data.y.cpu()
        y_pred = out.argmax(dim=-1, keepdim=True)
        
        test_acc = self.evaluator.eval({
            'y_true': y_true[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
            })['acc']
        
        train_acc = self.evaluator.eval({
            'y_true': y_true[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        val_acc = self.evaluator.eval({
            'y_true': y_true[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']

        return train_acc, val_acc, test_acc
    
    def _load_model(self):
        best_para, start_step= 0, 0
        if os.path.exists(self.pth_fn):
            checkpoint = torch.load(self.pth_fn)
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            self.model.load_state_dict(checkpoint['network_state_dict'])
            print(f'==> resuming from epoch {start_step}, best para {best_para:.4}')
        
        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None):
        save_fn=self.pth_fn if save_fn is None else save_fn
        torch.save({
            'step':step,
            'best_para':best_para,
            'network_state_dict': self.model.state_dict(),
        },save_fn)

    def _init_logger(self):
        self.logger = Logger(self.model_dir)

    def _log_data(self, results, step, prefix='train', verbose=False):
        log_results={}
        for k, v in results.items():
            if isinstance(v, float) or np.isscalar(v):
                log_results[k] = v
            elif type(v) == np.ndarray:
                log_results[k] = np.mean(v)
            else:
                log_results[k] = np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results, prefix, step, verbose)
        
        
        
