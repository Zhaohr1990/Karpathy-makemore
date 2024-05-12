""" 
This is my own implementation of model classes in makemore
Change the implementation of CharDataset so that we have chr_1, ... chr_{markov_order} -> chr_{markov_order + 1}
"""

# Import all necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# system inits (in case need to run script later)
torch.manual_seed(714)
torch.cuda.manual_seed_all(714)
random.seed(714)

# -----------------------------------------------------------------------------
# Ngram model
class Ngram(nn.Module):
    """
    Implementation of Ngram language model.
    """
    def __init__(self, vocal_size, markov_order):
        super().__init__()
        self.vocal_size = vocal_size # vocabulary size
        self.markov_order = markov_order # markov_order = 1 -> Bigram, markov_order = 2 -> Trigram
        self.layer = nn.Sequential(
            nn.Flatten(), # merge second and third dim
            nn.Linear((self.vocal_size+1) * self.markov_order, self.vocal_size)
        )

    def get_markov_order(self):
        return self.markov_order

    def forward(self, idx, targets=None):
        xenc = F.one_hot(idx, num_classes=self.vocal_size+1).float() # One hot encoding [b, l, markov_order, vocal_size+1]    
        logits = self.layer(xenc) # logits from linear layer

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets) # Cross entropy loss

        return logits, loss 

# -----------------------------------------------------------------------------
# MLP model
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
        self.layer = nn.Sequential(
            nn.Embedding(self.vocal_size + 1, self.emb_dim),
            nn.Flatten(), # merge second and third dim
            nn.Linear(self.emb_dim * self.markov_order, self.hid_dim), # plus 1 for leading <BLANK> token
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.vocal_size)
        )
        
    def get_markov_order(self):
        return self.markov_order

    def forward(self, idx, targets=None):
        
        logits = self.layer(idx) # logits from FF

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets) # Cross entropy loss

        return logits, loss 

# -----------------------------------------------------------------------------
# Helper function to generate samples and evaluate
@torch.no_grad()
def generate(model, dataset):
    model.eval()
    markov_order = model.get_markov_order()
    vocab_size = dataset.get_vocab_size()
    out = [vocab_size]*(markov_order-1) + [0]
    while True:
        logits, _ = model(torch.tensor([out])[:, -markov_order:])
        p = F.softmax(logits, dim=1)
        id = torch.multinomial(p, 1, replacement=True).item()
        out.append(id)
        if id == 0:
            break
    
    model.train()
    return(list(filter(lambda x: (x > 0) & (x < vocab_size), out))) # filtering out 0 and 27

# -----------------------------------------------------------------------------
# Helper function to create data loader
class CharDataset(Dataset):
    """
    Character-level data set based on inputted word list
    """
    def __init__(self, words, chars, max_word_length, markov_order):
        super().__init__()
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.markov_order = markov_order
        self.ctoi = {c:(i+1) for i, c in enumerate(self.chars)} # construct the character to integer mapping
        self.itoc = {i:c for c, i in self.ctoi.items()} # construct the integer to character mapping
        self._build_dataset()

    def __len__(self): # Need to modify
        return self.data.shape[0]
    
    def contains(self, word):
        return word in self.words
    
    def get_vocab_size(self):
        return len(self.chars) + 1
    
    def encode(self, word):
        return torch.tensor([self.ctoi[c] for c in list(word)], dtype=torch.long)
    
    def decode(self, l_pt):
        l_c = [self.itoc[i] for i in l_pt]
        return ''.join(l_c)
    
    def _build_word(self, word):
        wenc = self.encode(word)
        l = len(wenc)
        
        wenc = torch.cat((torch.tensor([self.get_vocab_size()]*(self.markov_order-1)), torch.tensor([0]), wenc, torch.tensor([0])))
        
        idx = torch.arange(0., len(wenc), dtype=torch.long)
        idx = idx.unfold(0, self.markov_order + 1, 1)

        return wenc[idx]

    def _build_dataset(self):
        wencs = []
        for word in self.words:
            wencs.append(self._build_word(word))

        self.data = torch.cat(wencs, dim=0)
    
    def __getitem__(self, index):
        data_sel = self.data[index, :]

        return data_sel[:-1], data_sel[-1]
    
def create_dataset(file_path, ratio=[0.1, 0.1], markov_order=3):
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
    
    val_dataset = CharDataset(val_words, chars, max_word_length, markov_order)
    test_dataset = CharDataset(test_words, chars, max_word_length, markov_order)
    train_dataset = CharDataset(train_words, chars, max_word_length, markov_order)

    return train_dataset, val_dataset, test_dataset, chars, max_word_length

def data_loader(dataset, batch_size=64): # flag = train

    dataloader = DataLoader(
        dataset,
        shuffle=True, 
        batch_size=batch_size,
        num_workers=0)
    
    return dataset, dataloader

# -----------------------------------------------------------------------------
# Helper function to evaluate
@torch.no_grad() # torch.inference_mode is preferable, but this is fine as long as no runtime error
def evaluate(model, dataset, batch_size=64, max_batches=None):
    model.eval()
    _, loader = data_loader(dataset, batch_size=batch_size)
    losses = []
    for i, (xspt, yspt) in enumerate(loader):
        logits, loss = model(xspt, yspt)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss
