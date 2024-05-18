import torch
import torch.nn.functional as F
import math
import random
import numpy as np
from makemore_origin import data_loader

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def count_params(model):
    """
    Count the parameters in a NN model: by layer and total number
    """
    out = {}
    for name, param in model.named_parameters():
        out[name] = sum(p.numel() for p in param.data)
    
    return out, sum(out.values())

def set_random_seed(seed):
    """
    Set random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# -----------------------------------------------------------------------------
# Helper function to generate samples and print 
@torch.no_grad()
def generate(model, train_dataset, batch_size, temperature=1, do_sample=True, top_k=None):
    model.eval()
    max_word_length = train_dataset.get_output_length()
    id = torch.zeros((batch_size, 1), dtype=torch.long)
    for _ in range(max_word_length - 1):
        logits, _ = model(id)
        logits = logits[:, -1, :].squeeze(1) / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        p = F.softmax(logits, dim=-1)

        if do_sample: 
            id_nxt = torch.multinomial(p, 1, replacement=True)
        else:
            _, id_nxt = torch.topk(p, k=1)

        id = torch.cat((id, id_nxt), dim=1)

    model.train()
    return id

def print_samples(id, train_dataset):
    id = id[:, 1:]
    for i in range(id.size(0)):
        row = id[i, :].tolist()
        end_idx = row.index(0)
        row = row[:end_idx]

        print(train_dataset.decode(row))

# -----------------------------------------------------------------------------
# Helper function for early stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('Inf')

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

# -----------------------------------------------------------------------------
# Helper function for adjusting learning rate
def get_lr(it, config):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# -----------------------------------------------------------------------------
# Helper function for configuring optimizer
def configure_optimizer(model, config):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer 
        optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate)

        return optimizer

# -----------------------------------------------------------------------------
# Helper function to evaluate
@torch.no_grad() # torch.inference_mode is preferable, but this is fine as long as no runtime error
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    _, loader = data_loader(dataset, batch_size)
    losses = []
    for i, (xspt, yspt) in enumerate(loader):
        logits, loss = model(xspt, yspt)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss