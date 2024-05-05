import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F

args = sys.argv
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open('tiny_shakespeare.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# NOTE(Abid): Mappping from characters to integers and vice versa
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    # NOTE(Abid): Setting the model to evaluation phase, just to infer
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    # NOTE(Abid): Setting the model to train phase
    model.train()

    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T(q), T(k))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v # (B, T(q), T(k)) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # Because of residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(n_embd, 4*n_embd),
                nn.ReLU(),
                nn.Linear(4*n_embd, n_embd), # Projection Layer, used because of the residual connection
                nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # NOTE(Abid): Communication
        self.sa = MultiHeadAttention(n_head, head_size)
        # NOTE(Abid): Computation
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # NOTE(Abid): each token would read a row of the embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # NOTE(Abid): idx and targets are both (B, T) tensor for integers
        tok_emb = self.token_embedding_table(idx) # This would be (B, T, C=n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # Notice the broadcast here, so it would be (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        # NOTE(Abid): Since cross_entropy requires the class indices (C) to be the 2nd dimension,
        #             we have to change the view of the data.
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # NOTE(Abid): idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # NOTE(Abid): So that the model doesn't consider more than block size
            idx_cond = idx[:, -block_size:]
            # NOTE(Abid): get the predictions
            logits, loss = self(idx_cond)
            # NOTE(Abid): focus only on the last time step
            logits = logits[:, -1, :] # This turns to (B, C)
            # NOTE(Abid): softmax for probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            # NOTE(Abid): sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # NOTE(Abid): append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = GPT()
m = model.to(device)

if len(args) > 1:
    print("Inference...")
    m.load_state_dict(torch.load(args[1]))
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=3000)[0].tolist()))
else:
    print("Training...")
    # NOTE(Abid): create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    
    for iter in range(max_iters):
    
        if iter % eval_interval == 0:
            losses = estimate_loss() # The base loss is ln(1/65) = 4.174
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
        # NOTE(Abid): sample a batch of data
        xb, yb = get_batch('train')
        
        # NOTE(Abid): evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # torch.save(model.state_dict(), 'shakespeare_gpt.pt')


"""
Example Output:

That which intirects me spakes to the hand,
Another I dream you; but that I may mean,
Last too much at you, such, swarm, as he first,
Achiefly, is it, like mine, so ruch as we
Praice to be soft-winged men's insumicst,
Stard to stone the issue of Hereford's sake;
The king is full of England ground Pilath,
The Valeria is obsequing;
But Roward; and thus hasty shortly for them;
Or for 'the shade of Hereford Richard's dearth;
Have I thwat, like safer an Edward mon,
The slanderous and triumphs, with sweet side,
To think the prince, thoroughly prick to my bed,
With daily shed block the man word.

FRIAR LAURENCE:
So place on his blood, the devil of his:
Saalm, divine jest, Prince Edward and Sir Fifth,
Well would repent in the war, to accurse here
Be here a man-placet, yet have to relieve;
Make hope more return.

FRATCLIFF:
I thank thee.

BENVOLIO:
O time; yet I come to such guilty tongue!

RICHMONTAGUE:
An heavens' sicker, it is, and, your honours ill:
The meaning both well use me: but being not that ill know
Requite might, I can love thee be to drink with thee.

TRANIO:
Come, little Rivers, and stand aloof;
Who now, by the dishonour of thee;
Which to reputation, if to this blest.

KING RICHARD II:
By my regal Sictlia daints,
Yet lieutenant in the leathe's right,
O, may it pity to about, you, I might,
Opening to the tribunes of your love's debt.

MONTAGUE:
Come, from Is shoulder boot in this business,
And thou seest she to please me close this spirit.
So, it were most chance that English rest
Be through the quarrell in just of her sheart;
That this exhales, the foe is not spring:
Yet nature is past; I unless
The procure of the that Romans, Lumposs are,
Horten's a man: the rest, as far from the death,
Most parasage is set a hopeful stick.

Lord:
The more of Isues for you.

First Servingman:
Are no other deed, and bestroke you on Master France?

Third Servant:
Nothing fellow's a newsman: till to the crown
How well is the contrary of the causes,
He should banish the clovers of Herefore have leave,
And heard you from your contract still-glass,
Disbate plebet; and if Edward's children,
His slep villain with the value, able,
Resign'd, and orpion'd years of dear your trial,
Which, howl's bittly, like most lightning sleep!

BUSHY:
Right wear it that Edward lives;
How sooneth may longest! I have kept a happy;
There words are all good no lesship and king:
Good hopes hence, be add indone, like simple.
Richard, O, dispatch, like the region for a course,
From redushing those two kings and sweet laws.

KING RICHARD II:
Either how thou canst by the corse.
I projure thee this deed mild,
Yet wash crown'd thou contradict it from so ease,
Who hast a Sureyor worthy grave's beauty,
A barbed both, the mildistress of safety,
Since he confounds me the boy the sea, that
'ere were as good weary he and legs that themselves are
To get the house.

LEONTES:

First Servingman:
My lord, my gorge for Reason;
Great king, go to send to the thereof.
So please you and your honours to see th
"""
