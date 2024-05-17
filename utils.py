import torch

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