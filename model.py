import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import math

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu

bleuscore_lst = []
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Use this instead of you want to use CPU
#device = torch.device('cpu')

f = open('Vol1-17.txt', encoding="utf8")
text = f.read() #reads the whole text as one string
f.close()

#build our vocabulary
vocab = list(set(text)) + ['<BOS>', '<EOS>', '<PAD>']
#since sets do not have ordering, without sorting, 
#vocab_stoi and vocab_itos are different each time
vocab = sorted(vocab) 
#build a dictionary mapping of word to unique index
vocab_itos = dict(enumerate(vocab)) 
#build a dictionary mapping of a unique index to a word (string)
vocab_stoi = {word:index for index, word in vocab_itos.items()} 
vocab_size = len(vocab)

texts = text.split("\n\n")
sentences = []#split the text into sentences
for text in texts:
    sentences += text.split("\n")


def make_samples(sizes):
  """
  creates samples of different sentence lengths from <sizes>.
  Will not create two samples with the same starting sentence.
  
  Parameters
    ----------
    sizes : list[int]
        a list of different sizes

    Returns
    -------
    samples of the text with different sentence lengths.
  
  """
    
    lst = []
    past = []
    for size in sizes:
        for i in range(0, len(sentences), size):
            used = False
            for num in past:
                if i % num == 0:
                    used = True
                    break
            if not used:
                lst.append("\n".join(sentences[i:i+size]))
        past.append(size)
    return lst

#make the data
data = make_samples([i for i in range(5, 9)])

#turn the data into lists of characters
data_ch = [["<BOS>"] + list(text) + ["<EOS>"] for text in data]
#turn the characters into ints
data_indices = [[vocab_stoi[ch] for ch in text] for text in data_ch]

class PositionalEncoding(nn.Module):
  """
    implementation from 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    modified so that the first dimension is the batch size
  """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe.to(device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)
    
class TransformerBlock(nn.Module):
  """
  follows transformer block from
  http://peterbloem.nl/blog/transformers.
  
  Uses pytorch's MultiheadAttention instead of the
  self attention module.
  """
  def __init__(self, k, heads):
    super().__init__()
    
    self.toqueries = nn.Linear(k, k, bias=False) 
    self.tokeys    = nn.Linear(k, k, bias=False)
    self.tovalues  = nn.Linear(k, k, bias=False)
    self.attention = nn.MultiheadAttention(k, heads, batch_first=True)
        
    self.norm1 = nn.LayerNorm(k)
    self.norm2 = nn.LayerNorm(k)

    self.ff = nn.Sequential(
      nn.Linear(k, 2 * k),
      nn.ReLU(),
      nn.Linear(2 * k, k))
    
  def forward(self, x):
      b, t, k = x.size()
      queries = self.toqueries(x)
      keys = self.tokeys(x)
      values = self.tovalues(x)
      
      #create attention mask to make model autoregressive
      indices = torch.triu_indices(t, t, offset=1)
      mask = torch.zeros(t,t, dtype=torch.bool, device=device).detach()
      mask[indices[0], indices[1]] = True
      
      attended = self.attention(queries, keys, values, attn_mask=mask)[0]
      x = self.norm1(attended + x)
      fedforward = self.ff(x)
      return self.norm2(fedforward + x)


class Transformer(nn.Module):
     """
  follows transformer from
  http://peterbloem.nl/blog/transformers.
  
  Uses positional encoding instead of positional embedding.
  
  """
    def __init__(self, k, heads, depth, vocab_size):
        
        super().__init__()

        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, k, padding_idx=vocab_stoi['<PAD>'])
        self.pos_encode = PositionalEncoding(k)

		# The sequence of transformer blocks that does all the 
		# heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

		# Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, vocab_size)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing 
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the 
                 classes (where c is the nr. of classes).
        """
        x.to(device)
		# generate token embeddings
        tokens = self.token_emb(x)
        b, t, k = tokens.size()
        
        x = self.pos_encode(tokens)
        x = self.tblocks(x)
        x = self.toprobs(x)
        return F.log_softmax(x, dim=1)
    
def train(model, batch_size=1, start_iter=0, start_epoch=1, num_epochs=1, lrate=0.001, wd=0,
          save_path = None):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    iters, losses = [], []
    #iters_sub, train_accs, val_accs  = [], [] ,[]
  
    model.train()
    counter = start_iter
    train_cost=-1
    for e in range(start_epoch, start_epoch+num_epochs): 
        np.random.shuffle(data_indices) #shuffle the data
        for i in range(0, len(data_indices)-batch_size, batch_size):
            batch = data_indices[i:i+batch_size] #this is the batch
            
            #make a list of tensors
            batch = [torch.t(torch.Tensor(text).long().unsqueeze(0)).to(device)
                for text in batch]
            #pad the tensors to get new tensor
            batch = nn.utils.rnn.pad_sequence(batch, True, vocab_stoi['<PAD>'])
            batch = batch.reshape(batch_size, batch.size(1))
            inp = batch[:,:-1] #<EOS> is not a input
            target = batch[:, 1:].detach() #<BOS> is not a target
            optimizer.zero_grad()
            output = model(inp)
            loss = criterion(output.reshape(-1, vocab_size),
                                    target.reshape(-1))  
            loss.backward()
            optimizer.step()
            train_cost = float(torch.Tensor.cpu(loss).detach().numpy())
            counter += 1
            
            if counter % 2000 == 0:
                iters.append(counter)
                losses.append(train_cost)
                if save_path is not None:
                    g = open(save_path[:-2].format('loss') + 'txt', 'a')
                else:
                    g = open('training_loss.txt', 'a')
                g.write(str(counter) +','+ str(train_cost) + '\n')
                g.close()
            #we delete tensors free free up memory
            del batch, inp, output, target, loss
            torch.cuda.empty_cache()
        
        print("Epoch %d. [Train Loss %f]" % 
              (e, train_cost))
        write_output(model, e + 1 + index)
        if save_path is not None:
                torch.save(model.state_dict(), save_path.format(e))
        
    return iters, losses, #iters_sub, train_accs, val_accs
        
def sample_sequence(model, max_len=5000, temperature=0.8):
    generated_sequence = ""
    
    model.eval()
    with torch.no_grad():
        inp = torch.Tensor([vocab_stoi["<BOS>"]]).long().unsqueeze(0)
        for p in range(max_len):
            inp = inp.to(device)
            output = model(inp)
            output = output[:, -1, :]
            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = int(torch.multinomial(output_dist, 1)[0])
            # Add predicted character to string and use as next input
            predicted_char = vocab_itos[top_i]
            if predicted_char == '<EOS>':
                break
        
            if predicted_char != '<PAD>' and predicted_char != '<BOS>':
                generated_sequence += predicted_char
                inp = inp.flatten().tolist()
                inp.append(top_i)
                inp = torch.Tensor(inp).long().unsqueeze(0)
    
    return generated_sequence

def remove_padding(s):
    x = s.replace('<PAD>', "")
    if len(x) == 0:
        return x
    if x[0] == " ":
        return x[1:]
    return x


def compute_bleu_score(model):
    #   sequence = remove_padding(generated_sequence)
    sequence = sample_sequence(model)
    sequence_lst = sequence.split("\n")
    s_removed = list(map(remove_padding, sequence_lst))
    s = list(map((lambda x: x.split(" ")), s_removed))
    n = len(s)
    i = 0
    res = 0
    cc = SmoothingFunction()
    while i + n < len(sentences):
        #   print("i : %d" % i)
        tmp = sentences[i:i + n]
        z = list(map((lambda x: x.split(" ")), tmp))
        score = corpus_bleu(s, z, smoothing_function=cc.method7)
        #   print("score is:%f " % score)
        res = max(res, score)
        bleuscore_lst.append(score)
        i += 1  # i += n use this if it's too slow
    print("max is %f" % res)


# list(map((lambda x: x.split(" ")),sentences))

def write_output(model, index):
    tmp_text = ""
    for _ in range(5):
        tmp_text += sample_sequence(model)
    f1 = open("model5/outputs/output_by5_%d.txt" % index, "w", encoding='utf-8')
    f1.write(tmp_text)
    f1.close()
    print("write output to: " + "model5/outputs/output_by5_%d.txt" % index)

	
model = Transformer(128, 8, 8, vocab_size)
#model.load_state_dict(torch.load('model[216,8,3]-40.pk'))
model.to(device)

path = 'model[128,8,8]-{}.pk'
info = train(model, batch_size=4, num_epochs=40, 
     lrate=0.0007, wd=0.001, 
     save_path=path)

plt.title("Learning Curve: Loss per Iteration")
plt.plot(info[0], info[1])
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

print(sample_sequence(model))
