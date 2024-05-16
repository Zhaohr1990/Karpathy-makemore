""" 
This is a reproduction of model classes in makemore
"""

# Import all necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math

# system inits
torch.manual_seed(714)
torch.cuda.manual_seed_all(714)
random.seed(714)

# -----------------------------------------------------------------------------
# Decoder language model
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention calculation based on Q, K, V matrices
    """
    def __init__(self, config):
        super().__init__()
        self.register_buffer("attn_mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, queries, keys, values):
        # Swap block_size and num_head, i.e., B, L, H, D -> B, H, L, D
        _, L, _, _ = queries.shape
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        attn = (queries @ keys.transpose(-2, -1)) / math.sqrt(keys.shape[-1]) # Scaled dot product between queries and keys
        attn_masked = attn.masked_fill(self.attn_mask[:, :, :L, :L] == 0, float('-inf')) # Masking
        p_masked = self.dropout(F.softmax(attn_masked, dim=-1)) # Softmax and dropout

        y = (p_masked @ values).transpose(1, 2).contiguous() # Weighted average of values

        return y
    
class AttentionLayer(nn.Module):
    """
    Multi-head attention layer under assumption that d_k = d_v = d_model
    """
    def __init__(self, config, attn_class):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.attn_class = attn_class
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.w_q = nn.Linear(self.d_model, self.d_model) # query projection
        self.w_k = nn.Linear(self.d_model, self.d_model) # key projection
        self.w_v = nn.Linear(self.d_model, self.d_model) # value projection
        self.c_proj = nn.Linear(self.d_model, self.d_model) # projection after concat
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x) # generate q, k, v matrices
        q = q.view(B, L, self.n_head, self.d_model // self.n_head) # multi-head
        k = k.view(B, L, self.n_head, self.d_model // self.n_head)
        v = v.view(B, L, self.n_head, self.d_model // self.n_head)

        y = self.attn_class(q, k, v).view(B, L, -1) # multi-head attention
        y = self.dropout(self.c_proj(y)) # projection after concat and dropout 

        return y
    
class DecoderLayer(nn.Module):
    """
    Layernorm1 -> Attention layer -> Layernorm2 -> FF (skip connection)
    """
    def __init__(self, config, attn_layer):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(config.d_model)
        self.attn_layer = attn_layer
        self.layernorm2 = nn.LayerNorm(config.d_model)
        self.ln_layer1 = nn.Linear(config.d_model, config.d_ff)
        self.activation = F.gelu
        self.ln_layer2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.attn_layer(self.layernorm1(x)) # attention layer with layernorm, dropout added in attention layer 
        x = x + self.dropout(self.ln_layer2(self.activation(self.ln_layer1(self.layernorm2(x))))) # FF layer with layernorm

        return x
    
class Decoder(nn.Module):
    """
    Implementation of decoder model.
    """
    def __init__(self, config, vocal_size):
        super().__init__()
        self.vocal_size = vocal_size
        self.block_size = config.block_size
        self.tok_emb_layer = nn.Embedding(self.vocal_size, config.d_model) # token embedding
        self.pos_emb_layer = nn.Embedding(config.block_size, config.d_model) # positional embedding
        self.layers = nn.ModuleList([
            DecoderLayer(config, AttentionLayer(config, MultiHeadAttention(config))) for _ in range(config.n_layer)
        ])
        self.lm_head = nn.Linear(config.d_model, self.vocal_size, bias=False)
        
    def get_markov_order(self):
        return self.block_size

    def forward(self, idx, targets=None):
        B, L = idx.shape

        x = self.tok_emb_layer(idx) + self.pos_emb_layer(torch.arange(0, L).unsqueeze(0))
        for block in self.layers:
            x = block(x) 
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocal_size), targets.view(-1), ignore_index=-1) # Cross entropy loss

        return logits, loss 

# -----------------------------------------------------------------------------
# Bigram language model
class Bigram_naive(nn.Module):
    """
    Naive implementation of bigram language model. 
    Define logits weight matrix 
    Explicit logsoftmax + negative likelihood loss
    """
    def __init__(self, vocal_size):
        super().__init__()
        self.vocal_size = vocal_size
        self.w = nn.Parameter(torch.zeros((self.vocal_size, self.vocal_size)))

    def get_markov_order(self):
        return 1

    def forward(self, idx, targets=None):
        logits = self.w[idx] # logits by look up 
        p = F.softmax(logits, dim=-1) # softmax to probabilities
        p = p.view(-1, self.vocal_size) # reshaping to 2D tensor
        targets = targets.view(-1) # reshaping to 1D tensor

        p = p[targets != -1, :] # filtering out -1
        targets = targets[targets != -1] # filtering out -1

        loss = None
        if targets is not None:
            loss = -p[torch.arange(len(targets)), targets].log().mean() # Negative likelihood loss

        return logits, loss 

class Bigram_crossentropy(nn.Module):
    """
    Naive implementation of bigram language model with the use of cross entropy loss.
    Define logits weight matrix 
    Cross entropy loss = logsoftmax + negative likelihood loss 
    """
    def __init__(self, vocal_size):
        super().__init__()
        self.vocal_size = vocal_size
        self.w = nn.Parameter(torch.zeros((self.vocal_size, self.vocal_size)))
    
    def get_markov_order(self):
        return 1

    def forward(self, idx, targets=None):
        logits = self.w[idx] # logits by look up 

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocal_size), targets.view(-1), ignore_index=-1) # Cross entropy loss

        return logits, loss 
    
class Bigram(nn.Module):
    """
    Implementation of bigram language model with the use of cross entropy loss.
    Define a linear layer to map one-hot encoding to logits
    Cross entropy loss 
    """
    def __init__(self, vocal_size):
        super().__init__()
        self.vocal_size = vocal_size
        self.w_layer = nn.Linear(self.vocal_size, self.vocal_size)

    def get_markov_order(self):
        return 1

    def forward(self, idx, targets=None):
        xenc = F.one_hot(idx, num_classes=self.vocal_size).float() # One hot encoding 
        logits = self.w_layer(xenc) # Linear layer mapping to logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocal_size), targets.view(-1), ignore_index=-1) # Cross entropy loss

        return logits, loss 

class Ngram(nn.Module):
    """
    Implementation of Ngram language model.
    """
    def __init__(self, vocal_size, markov_order):
        super().__init__()
        self.vocal_size = vocal_size # vocabulary size
        self.markov_order = markov_order # markov_order = 1 -> Bigram, markov_order = 2 -> Trigram
        self.w_layer = nn.Linear((self.vocal_size+1) * self.markov_order, self.vocal_size) # plus 1 for leading <BLANK> token

    def get_markov_order(self):
        return self.markov_order

    def forward(self, idx, targets=None):
        b, l = idx.shape
        idx = idx.unsqueeze(2) # adding 1 dimension for cancat
        
        # <BLANK>, 0 -> chr 1; 0, chr1 -> chr 2; ...
        embs = []
        for _ in range(self.markov_order):
            embs.append(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocal_size 
        embs = list(reversed(embs))
        
        x = torch.cat(embs, -1) # [b, l, markov_order]
        xenc = F.one_hot(x, num_classes=self.vocal_size+1).float() # One hot encoding [b, l, markov_order, vocal_size+1]    
        xenc = xenc.view(b, l, -1) # [b, l, (vocal_size+1)*markov_order] 

        logits = self.w_layer(xenc) # logits from linear layer

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocal_size), targets.view(-1), ignore_index=-1) # Cross entropy loss

        return logits, loss 

class MLP(nn.Module):
    """
    Implementation of MLP language model.
    """
    def __init__(self, vocal_size, markov_order, emb_dim, hid_dim):
        super().__init__()
        self.vocal_size = vocal_size # vocabulary size
        self.markov_order = markov_order # similar to Ngram
        self.emb_dim = emb_dim # embedding dimension
        self.hid_dim = hid_dim
        self.embed_layer = nn.Embedding(self.vocal_size + 1, self.emb_dim)
        self.layer = nn.Sequential(
            nn.Linear(self.emb_dim * self.markov_order, self.hid_dim), # plus 1 for leading <BLANK> token
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.vocal_size)
        )
        
    def get_markov_order(self):
        return self.markov_order

    def forward(self, idx, targets=None):
        
        # <BLANK>, 0 -> chr 1; 0, chr1 -> chr 2; ...
        embs = []
        for _ in range(self.markov_order):
            emb = self.embed_layer(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocal_size 
            embs.append(emb)
        embs = list(reversed(embs))
        
        x = torch.cat(embs, -1) # [b, l, markov_order * emb_dim]
        logits = self.layer(x) # logits from FF

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocal_size), targets.view(-1), ignore_index=-1) # Cross entropy loss

        return logits, loss 

# -----------------------------------------------------------------------------
# Helper function to generate samples and evaluate
@torch.no_grad()
def generate(model):
    model.eval()
    out = [0]
    while True:
        logits, _ = model(torch.tensor(out).unsqueeze(0))
        logits = logits[:, -1, :]
        p = F.softmax(logits, dim=1)
        id = torch.multinomial(p, 1, replacement=True).item()
        out.append(id)
        if id == 0:
            break
    
    model.train()
    return(list(filter(lambda x: x > 0, out))) # filtering out 0

# -----------------------------------------------------------------------------
# Helper function to create data loader
class CharDataset(Dataset):
    """
    Character-level data set based on inputted word list
    """
    def __init__(self, words, chars, max_word_length):
        super().__init__()
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.ctoi = {c:(i+1) for i, c in enumerate(self.chars)} # construct the character to integer mapping
        self.itoc = {i:c for c, i in self.ctoi.items()} # construct the integer to character mapping

    def __len__(self):
        return len(self.words)
    
    def contains(self, word):
        return word in self.words
    
    def get_vocab_size(self):
        return len(self.chars) + 1
    
    def get_output_length(self):
        return self.max_word_length + 1 # <START> <END> token wrapping words
    
    def encode(self, word):
        return torch.tensor([self.ctoi[c] for c in list(word)], dtype=torch.long)
    
    def decode(self, l_pt):
        l_c = [self.itoc[i] for i in l_pt]
        return ''.join(l_c)
    
    def __getitem__(self, index):
        word = self.words[index]
        wenc = self.encode(word)
        l = len(wenc)

        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)

        x[1:(l+1)] = wenc
        y[:l] = wenc
        y[(l+1):] = -1 # Masking 

        return x, y
    
def create_dataset(file_path, ratio=[0.1, 0.1]):
    """
    This is the function to read txt file, split lines, perform train/val/test split on word level
    """
    words = open(file_path, 'r').read().splitlines() # load the data (can add pre-processing steps)
    chars = sorted(set(''.join(words))) # get unique characters 
    random.shuffle(words) # randomly schuffle the words
    max_word_length = max([len(list(w)) for w in words])

    # can assert the sum of ratios
    nwords = len(words)
    nval = int(nwords * ratio[0])
    ntest = int(nwords * ratio[1])

    val_words, test_words, train_words = words[:nval], words[nval:(nval+ntest)], words[(nval+ntest):]
    
    val_dataset = CharDataset(val_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)
    train_dataset = CharDataset(train_words, chars, max_word_length)

    return train_dataset, val_dataset, test_dataset, chars, max_word_length

def data_loader(dataset, batch_size=50): # flag = train

    dataloader = DataLoader(
        dataset,
        shuffle=True, 
        batch_size=batch_size,
        num_workers=0)
    
    return dataset, dataloader

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
