import torch
import random

@torch.no_grad()
def inference(model, loader, depths_list, device):
    model.eval()
    inference_depths = []
    xs = []
    ids = []

    for batch in loader:
        ids = ids + batch.n_id[:batch.batch_size].tolist()
        depth = random.choice(depths_list)
        inference_depths = inference_depths + [depth]*batch.batch_size 
        out = model.forward(batch.x.to(device), batch.edge_index.to(device), depth)
        out = out[:batch.batch_size]
        xs.append(out.cpu())

    out_all = torch.cat(xs, dim=0)

    return out_all, inference_depths, ids