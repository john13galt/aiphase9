import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from safetensors import safe_open
from ollama import chat
from transformers import AutoTokenizer

# import torchvision
import time
# from torchvision.datasets import MNIST
# from torchvision import transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
import numpy as np

# new positional info functions
# First calculate angels, then rotate
def rope_angles(position, embedsize):
    # Positions is a 1-D tensor.  Embedsize is an interger
    assert embedsize % 2 == 0, f"embedsize is odd: {embedsize}"
    # first make angles shape (embedsize//2, len(positions)
    angles = torch.zeros((int(embedsize/2),len(position)), device=position.device)
    for i in range(len(angles)):
        angles[i] = position / 10000 ** (2*i / embedsize)
    # transpose because angles should be shape (len(position), embedsize//2)
    return angles.transpose(0,1)

def rotate(x, theta):
    # x is a n-D tensor of embeddings of (batchsize, headcount, sequence_length, embedsize)
    #   but batchsize & headcount might be missing!  
    # theta is a 2-D tensor of rotations (sequence_length, embedsize//2)
    assert x.shape[-1] % 2 == 0, f"x.shape[-1] length is odd {x.shape[-1]}"
    assert x.shape[-1] == 2*theta.shape[1], f"theta must be half the size of x's last dimension"
    assert x.shape[-2] == theta.shape[0], f"sequence length mismatch between x and theta"
    originalshape = x.shape
    # first split the last dimension only.  embedsie --> (embedsize//2,)
    # the "*" does "star unpacking".  If orginalshape is (28,2,6), then i creates 28, 2, 6 as separate arguments for reshape
    # this should give me (batchsize, headcount, sequence_length, headsize,2)
    #   though batchsize & headcount may not exist
    x = x.reshape(*originalshape[:-1], -1, 2)
    # now I need to split it into two (batchsize, headcount, sequence_length, headsize) tensors
    x0, x1 = torch.unbind(x, dim=-1)
    # each newpos will be (m, sequence_length, embedsize//2)... hope that the multiplication broadcasts
    newpos0 = x0 * torch.cos(theta) - x1 * torch.sin(theta)
    newpos1 = x0 * torch.sin(theta) + x1 * torch.cos(theta)
    # final output shape must match originalshape
    return torch.stack((newpos0, newpos1), dim=-2).reshape(originalshape)

# TransformerBlock that works with batches
class TransformerBlock(nn.Module):
    def __init__(self, embedsize, headcount, kv_headcount):
        super().__init__()
        self.headcount = headcount
        self.kv_headcount = kv_headcount
        self.kv_group_size = headcount//kv_headcount
        assert headcount % kv_headcount == 0, f"headcount & kv_headcount mismatch"
        self.headsize = embedsize//headcount
        self.norm1 = nn.LayerNorm(embedsize)
        # This time make weights as if there is only 1 head and make them a Linear layer
        self.WQ = nn.Linear(embedsize, embedsize, bias=False)
        self.WK = nn.Linear(embedsize, embedsize//self.kv_group_size, bias=False)
        self.WV = nn.Linear(embedsize, embedsize//self.kv_group_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.attentionoutput = nn.Linear(embedsize, embedsize, bias=False)
        self.norm2 = nn.LayerNorm(embedsize)
        # arbitrary... made my FFN expand to 2x embedsize.  Could be bigger.
        self.ffnexpand = nn.Linear(embedsize, 2 * embedsize)
        self.relu = nn.ReLU()
        self.ffncontract = nn.Linear(2 * embedsize, embedsize)

    def forward(self, x, cache, position):
        # x.shape = (batchsize, inputsize, embedsize)
        # Normalize (scale based on weights).  xnorm stays = (batchsize, inputsize, embedsize)
        xnorm = self.norm1(x)
        # Make Q, K, V vectors.  
        # (batchsize, inputsize, embedsize) x (embedsize, embedsize) = (batchsize, inputsize, embedsize)
        Q = self.WQ(xnorm)
        K = self.WK(xnorm)
        V = self.WV(xnorm)
        # But after I create the vectors, I split them into multi-heads ... first reshape to split the last dimension
        # Note... here is where I'm doing DQA with fewer heads for K & V
        Q = Q.view(x.shape[0], x.shape[1], self.headcount, self.headsize)
        K = K.view(x.shape[0], x.shape[1], self.kv_headcount, self.headsize)
        V = V.view(x.shape[0], x.shape[1], self.kv_headcount, self.headsize)
        # Then swap dimensions to put heads first, makes it (batchsize, headcount, inputsize, headsize)
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        # Now, here we diverge.  
        #       If there is no cache, operate normally (and create the cache) - training or first inference pass
        #       If there is cache, use the KV cache and only do the new input
        if not(cache):
            # This is where the new RoPE positional stuff goes. positions shape is 1D (inputsize,)
            positions = torch.arange(x.shape[1], device=Q.device)
            # theta shape is 2d (inputsize, headsize//2)
            theta = rope_angles(positions,Q.shape[-1])
            # now I have to rotate Q and K shaped (batchsize, headcount, inputsize, headsize)
            Q = rotate(Q, theta)
            K = rotate(K, theta)
            # First, store K & V in a cache. Cache is a dict with K anv V members
            # Each member is a tensor fo shape (batchsize, headcount, sequence_so_far, headsize)
            # sequence_so_far must stay less than context window, but let generator handle that
            # Note:  K & V have kv_headcount which is < regular headcount, to save memory (DQA)
            newcache = {}
            newcache['K'] = K
            newcache['V'] = V
            cache = newcache
            # Now I need to expand the K and V vectors so they match Q
            #       This way uses more memory, but at least the cache stays smaller
            K = torch.repeat_interleave(K, repeats=self.headcount//self.kv_headcount, dim=1)
            V = torch.repeat_interleave(V, repeats=self.headcount//self.kv_headcount, dim=1)
            # This scaled_dot_product_attention function automatically does all the stuff I'm commenting out:
            # Calculate scores (Q x K.T), mask it, and apply softmax to get Attention
            # then outputs = Attention x V
            outputs = F.scaled_dot_product_attention(Q,K,V, is_causal=True)
            '''    
            # Create scores with Q x K.T - automatically works for batches
            # (batchsize, headcount, inputsize, headsize) x (batchsize, headcount, headsize, inputsize)
            scores = (Q @ K.transpose(-1, -2)) / (self.headsize ** 0.5)
            # the [-2:] means it will work no matter how many batches or heads
            mask = torch.tril(torch.ones(scores.shape[-2],scores.shape[-1], dtype=torch.bool, device = scores.device)) 
            # apply mask (different syntax than numpy) - automatically works for batches
            scores = scores.masked_fill(mask == 0, -torch.inf)
            # softmax layer creates Attention weights . works with batches since created with "dim=-1"
            A = self.softmax(scores) # dim=-1 in the layer, so not needed here
            # Works with batches:  (batchsize, headcount, inputsize, inputsize) x (batchsize, headcount, inputsize, headsize)
            outputs = A @ V # creates (batchsize, headcount, inputsize, headsize)
            '''
        else: # there is a cache... and original x's seqence_length must = 1
            # This is where the new RoPE positional stuff goes. positions shape is 1D (inputsize,)
            positions = torch.tensor([position], device=Q.device)
            # theta shape is 2d (inputsize, headsize//2)
            theta = rope_angles(positions,Q.shape[-1])
            # now I have to rotate Q and K shaped (batchsize, headcount, inputsize, headsize)
            Q = rotate(Q, theta)
            K = rotate(K, theta)
            # I need to append the new K and V to the existing cache
            # existing cache is size (1, kv_headcount, sequence_length_so_far, headsize)
            # newcache is (1, kv_headcount, 1, headsize)
            # Appended, becomes (1, headcount, sequence_length_so_far+1, headsize)
            cache['K'] = torch.cat((cache['K'], K), dim=2)
            cache['V'] = torch.cat((cache['V'], V), dim=2)
            # Now I need to expand the K and V vectors in the cache so they match Q
            #       This way uses more memory, but at least the cache stays smaller
            # Note that I use temporary variables so I don't expand the actual cache
            tempKcache = torch.repeat_interleave(cache['K'], repeats=self.headcount//self.kv_headcount, dim=1)
            tempVcache = torch.repeat_interleave(cache['V'], repeats=self.headcount//self.kv_headcount, dim=1)
            # This scaled_dot_product_attention function automatically does all the stuff I'm commenting out
            # Calculates scores (Q x K.T), softmax to get attention, and outputs = Attention x V
            # Note that no mask is needed because caching is used only when x's sequence length = 1
            outputs = F.scaled_dot_product_attention(Q,tempKcache,tempVcache, is_causal=False)
            ''' 
            # scores becomes (1, headcount, 1, sequence_length_so_far+1)
            scores = (Q @ tempKcache.transpose(-1, -2)) / (self.headsize ** 0.5)
            # softmax layer creates Attention weights . works with batches since created with "dim=-1"
            A = self.softmax(scores) # dim=-1 in the layer, so not needed here
            # (1, headcount, 1, sequence_length_so_far+1) x (1, headcount, sequence_length_so_far+1, headsize)
            outputs = A @ tempVcache # creates (1, headcount, 1, headsize)
            '''
        # Concatenate the multiple heads back together
        # first transpose to (batchsize, inputsize, headcount, headsize), then reshape the last two dimensions to one
        concat = outputs.transpose(1,2).reshape(x.shape[0],x.shape[1],-1)
        # Run through final weights layer
        attentionoutputs = self.attentionoutput(concat)
        # Residual Connection
        outputs = attentionoutputs + x
        # Normalize again
        xnorm = self.norm2(outputs)
        # Feed Forward network - expand, activate w/ ReLU, then contract
        expand = self.ffnexpand(xnorm)
        active = self.relu(expand)
        contract = self.ffncontract(active)
        # and do another resitual connection
        finaloutput = contract + outputs
        return finaloutput, cache

class TransformerModel(nn.Module):
    def __init__(self, vocabsize, embedsize, headcount, kv_headcount, blockcount):
        super().__init__()
        self.embedsize = embedsize
        self.headsize = embedsize//headcount
        self.headcount = headcount
        self.kv_headcount = kv_headcount
        self.E = nn.Embedding(vocabsize, embedsize)
        # ModuleList is how I create the multiple sequential transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(embedsize, headcount, kv_headcount) for _ in range(blockcount)])
        self.norm = nn.LayerNorm(embedsize)
        self.output = nn.Linear(embedsize, vocabsize)

    def forward(self, token_ids, cache=None, position=0):
        # token_ids is (batchsize, inputsize), though batchsize, and inputsize can be 1
        # New forward that accepts an external KV cache... so it must return a KV cache 
        # Do not pass in a cache when doing training
        # Do not pass in a cache when processing the prompt (first step in autoregressive inference)
        # Pass in a cache when generating all subseqent tokens one at a time (inputsize = 1)
        # First apply embeddings to tokens. creates (batchsize, inputsize, embedsize)
        x = self.E(token_ids)
        # go through Transformer blocks, send a cache if there is one
        newcache = [None] * blockcount
        for block, i in zip(self.blocks, range(blockcount)):
            # have to send in the cache and get it out
            if cache:
                x, blockcache = block(x, cache[i], position)
            else:                
                x, blockcache = block(x, None, 0)
            newcache[i] = blockcache
        # normalize output (batchsize, inputsize, embedsize)
        x = self.norm(x)
        # final output layer creates (bastchsize, inputsize, vocabsize)
        output = self.output(x)
        # have to return the cache also now
        return output, newcache

# define parameters
vocabsize = 6
inputsize = 6
embedsize = 8
headsize = 4
headcount = embedsize//headsize # must be an integer
kv_headcount = headcount//2
ffnsize = 2 * embedsize
blockcount = 2

# this is important... allows me to use the GPU.  Use "device" when creating tensors
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Create my training data set.  This is kind of fake... each word is a token (not using tokenizer)
text = """
the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran up the hill
the cat ran up the hill
"""

# split text into words (for now, words = tokens)
tokens = text.lower().split()
# get only unique words to create vocab (just a list).  Doesn't have to be sorted, but we are
vocab = sorted(set(tokens))
vocabsize = len(vocab)
# create dicts to turn tokens to indexes and vice-versa
token_to_id = {token: i for i, token in enumerate(vocab)}
id_to_token = {i: token for token, i in token_to_id.items()}

# Turn original text tokens into token_ids for training
token_ids = torch.tensor([token_to_id[token] for token in tokens], dtype = torch.long, device=device)

# create a "batch" (inputs & targets) of all 6-word training sequences using the text defined above
batchsize = len(token_ids) - inputsize
inputs = []
targets = []
for i in range(batchsize):
    # inputs are "inputsize" sequences of token_ids
    input_sequence = token_ids[i : i + inputsize]
    # targets are "inputsize" sequences of the immediate next token_ids (shifted one)
    target_sequence = token_ids[i + 1 : i + inputsize + 1]

    inputs.append(input_sequence)
    targets.append(target_sequence)
# torch.stack takes a bunch of individual tensors and makes them into one big tensor
inputs = torch.stack(inputs)
targets = torch.stack(targets)
# move data to CPU/GPU
inputs = inputs.to(device)
targets = targets.to(device)

# NOTE:  Important - Build model & move it to device
model = TransformerModel(vocabsize, embedsize, headcount, kv_headcount, blockcount)
model = model.to(device)

# Get ready to train
model.train()
# use built in Adam optimizer function to update model parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# NOTE:  Important - Training loop.  Used an arbitrary # of training epochs, 1 batch per epoch
for epoch in range(1000): 
    # model forward pass on a single batch of all the training data at once.  Ignore the cache
    logits, _ = model(inputs)
    # calculate loss ... F = torch.nn.functional
    loss = F.cross_entropy(logits.reshape(-1,vocabsize), targets.reshape(-1))
    # gclear out old gradients, calculate new ones with .backward(), and update parameters with .step()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # check that loss is getting better
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, loss: {loss}")


print("\n")
print("--------------------------------------------------------")
print("-"*10, "Lesson 38.x - Byte Pair Encoding (BPE) Tokenizer", "-"*10)
print("--------------------------------------------------------")

import re
import string

def make_tokenlist(text):
    tokenlist = []
    # split everything into words (including punctuation), but strips the spaces
    pattern = r"\w+|[^\w\s]"
    words = re.findall(pattern, text.lower())
    # split words into letters... with a space before words and a null char at the end
    # punctuation only gets the null character at the end.
    for word in words:
        if word not in string.punctuation:
            # this is the magic:  list(text) splits the text inot a list of individual characters
            tokenlist += list(' ' + word + '\0')
        else:
            tokenlist += list(word + '\0')
    return tokenlist

def get_pairs(tokenlist):
    # go through tokenlist and create all possible pairs of tokens (which ar strings)
    # so if the token list is ['a','b','c'], it returns ["ab", "bc"]
    pairs = []
    length = len(tokenlist)
    # loop through each token
    for i in range(length):
        # make sure there's a next token and the end of the current token isn't the end of a word
        if i < length - 1 and tokenlist[i][-1] != '\0':
            pairs.append(tokenlist[i] + tokenlist[i+1])
        else:
            pairs.append(tokenlist[i])
    return pairs

def best_pair(pairs, vocab):
    # go through the list of pairs and count how often each occurs
    # return the one that occus the most (and the number of occurences)
    counts = {}
    for pair in pairs:
        if pair == '\0': continue # skip end of words
        if pair in counts:
            counts[pair] += 1
        else:
            counts[pair] = 1
    most = 0
    mostpair = ''
    for item in counts:
        if counts[item] > most and item not in vocab:
            most = counts[item]
            mostpair = item
    return mostpair, most

def merge_pairs(tokenlist, best):
    # look through the token list and merge all occurences of the best pair
    # so if best is "ab", look for all the 'a', 'b' occurences and replace them with "ab"
    # there's a reason we go through the list backwards to do it, but I don't remember why :-)
    length = len(tokenlist)
    for i in range(length-2, -1, -1): # go through the list backwards, starting at the next to last item
        if best == tokenlist[i] + tokenlist[i+1]:
            tokenlist[i] = best
            tokenlist.pop(i+1)
    return tokenlist


# this text is what I'm going to tokenize... it's not what I trained the model on above
text = '''
 Eddie Willers walked on, wondering why he always felt it at this time of day, this sense of dread without reason. No, he thought, not dread, there’s nothing to fear: just an immense, diffused apprehension, with no source or object. He had become accustomed to the feeling, but he could find no explanation for it; yet the bum had spoken as if he knew that Eddie felt it, as if he thought that one should feel it, and more: as if he knew the reason.
Eddie Willers pulled his shoulders straight, in conscientious self-discipline. He had to stop this, he thought; he was beginning to imagine things. Had he always felt it? He was thirty-two years old. He tried to think back. No, he hadn’t; but he could not remember when it had started. The feeling came to him suddenly, at random _intervals, and now it was coming more often than ever. It’s the twilight, he thought; I hate the twilight.
The clouds and the shafts of skyscrapers against them were turning brown, like an old painting in oil, the color of a fading masterpiece. Long streaks of grime ran from under the pinnacles down the slender, soot-eaten walls. High on the side of a tower there was a crack in the shape of a motionless lightning, the length of ten stories. A jagged object cut the sky above the roofs; it was half a spire, still holding the glow of the sunset; the gold leaf had long since peeled off the other half. The glow was red and still, like the reflection of a fire: not an active fire, but a dying one which it is too late to stop.
No, thought Eddie Willers, there was nothing disturbing in the sight of the city. It looked as it had always looked.
He walked on, reminding himself that he was late in returning to the office. He did not like the task which he had to perform on his return, but it had to be done. So he did not attempt to delay it, but made himself walk faster.
He turned a corner. In the narrow space between the dark silhouettes of two buildings, as in the crack of a door, he saw the page of a gigantic calendar suspended in the sky.
It was the calendar that the mayor of New York had erected last year on the top of a building, so that citizens might tell the day of the month as they told the hours of the day, by glancing up at a public tower. A white rectangle hung over the city, imparting the date to the men in the streets below. In the rusty light of this evening’s sunset, the rectangle said: September 2.
Eddie Willers looked away. He had never liked the sight of that calendar. It disturbed him, in a manner he could not explain or define. The feeling seemed to blend with his sense of uneasiness; it had the same quality.
He thought suddenly that there was some phrase, a kind of quotation, that expressed what the calendar seemed to suggest. But he could not recall it. He walked, groping for a sentence that hung in his mind as an empty shape. He could neither fill it nor dismiss it. He glanced back. The white rectangle stood above the roofs, saying in immovable finality: September 2.
'''

# set a maximum vocabsize.  tokenize will stop at best token representation or at this limit
vocabsize = 1000

# strategy is to take token list which starts as individual characters, and combine them into 
# every possible side by side pair (without going over word boundaries).  Then count the occurrence
# of those pairs in the training text.  Find the pair that occurs the most and add it to the vocabulary
# then "combine" all instances of that pair in the token list, and repeat
# stop when the you're combining just single occurence pairs or you hit the max vocab size
def tokenize(text, maxtokens):
    # create initial vocab by turning the entire training text in a list of unique characters
    # so my vocab starts with 1 character tokens
    # I'm forcing everyting to lowercase for simplicity for now
    vocab = list(set(list(text.lower()))) 
    # Now 
    tokenlist = make_tokenlist(text)
    # print(tokenlist)
    DEBUG = False
    while len(vocab) < vocabsize:
        if DEBUG: print('----- ', i, ' -----')
        # 
        pairs = get_pairs(tokenlist)
        if DEBUG: print(pairs)
        best, count = best_pair(pairs, vocab)
        if DEBUG: print(best, '-', count)
        if count == 1: break
        tokenlist = merge_pairs(tokenlist, best)
        if DEBUG: print(tokenlist)
        vocab = [best] + vocab
        if DEBUG: print(vocab)
    return vocab, tokenlist

vocab, tokenlist = tokenize(text, vocabsize)
print(f"Final vocab size: {len(vocab)}\n{vocab}")
print()
print(f"Final tokenlist: {tokenlist}")

print("\n")
print("--------------------------------------------------------")
print("-"*10, "Lesson 40 - Continuous token generation", "-"*10)
print("-"*10, "Lesson 41 - Sampling & Temperature", "-"*10)
print("-"*10, "Lesson 44 - KV Cache", "-"*10)
print("--------------------------------------------------------")
# Uses model trained in Lesson 38/39

def print_sequence(token_ids):
    for item in token_ids:
        print(f"{id_to_token[item.item()]} ", end = '')
    print("\n")

def get_topk(logits, k):
    kth = torch.kthvalue(logits, k = len(logits) - k + 1).values.item()
    logits[logits < kth] = -torch.inf
    return logits

def get_topp(logits, p):
    probabilities = F.softmax(logits, dim=-1)
    sorted_logits, _ = torch.sort(probabilities, dim=-1, descending=True)
    count = 0
    sum = 0
    while sum < p:
        sum += sorted_logits[count]
        count += 1
    return get_topk(logits, count)

def get_token_from_logits(logits, param, temp):
    if param:
        if temp:
            logits /= temp
        if param >= 1:
            logits = get_topk(logits, param)
        else:
            logits = get_topp(logits, param)
        probabilities = F.softmax(logits, dim=-1)
        # don't need to unsqueeze when I use multinomial
        new_token = torch.multinomial(probabilities, num_samples=1)
    else:
        new_token = torch.argmax(logits).unsqueeze(0)
    return new_token

# after training, use this to generator tokens
def generator(token_ids, tokens_to_generate, context_window, model, param=0, temp=0):
    # Feed in a few starter tokens... token_ids is shape (sequencelen)
    # tokens to generate is just an integer
    # context window is an integer
    # Batch is always 1 when generating, sequencelen can be anything < context_widow
    # NOTE:  Does not always pick the best token... uses probabilities based on param and temp
    #   if param skipped... use greedy
    #   if param < 1, use top-p (choose from tokens that make up the top 'p' (a percentage) next token options)
    #   if param > 1, use top-k (choose from the top 'p' (a count)) of token options)
    #   apply temp if included for top-p/top-k.  
    #       if temp is < 1, will amplify token probability differences, make it more likely to pick only the best
    #       if temp is > 1, will shrink token probability differences, make it more likely to pick alternative options
    startlen = len(token_ids)
    generated_tokens = token_ids
    # make sequence (1, sequencelen)... because model requires 2D input
    sequence = token_ids.unsqueeze(0)
    
    with torch.no_grad():
        # first pass, use entire input token_ids... this is the "prompt", create the KV Cache
        # model returns shape=(bastchsize, inputsize, vocabsize), and kvcache (as a tuple)
        # with [0][-1], logits is just (vocabsize)
        response = model(sequence) # (batch = 1)
        # break the response in the returned logits and kvcache
        logits = response[0][0][-1]
        kvcache = response[1]
        # need next_position to do the RoPE positional encoding correctly
        next_position = kvcache[0]['K'].shape[-2]
        # probabilistically get the next token to use
        new_token = get_token_from_logits(logits, param, temp)
        # and add it to the list of tokens I've generated
        generated_tokens = torch.cat((generated_tokens, new_token), dim = 0)
        # print_sequence(generated_tokens)
        # make sure kvcahce doesn't get longer than context window
        for blockcache in kvcache:
            blockcache['K'] = blockcache['K'][:, :, -context_window:, :]
            blockcache['V'] = blockcache['V'][:, :, -context_window:, :]
        # remaining passes, use only new_token, use the KV Cache
        while (len(generated_tokens) - startlen) < tokens_to_generate:
            # run the model and get the next logits and updated kvcache.  have to tell it position for these passes
            logits, kvcache = model(new_token.unsqueeze(0), cache=kvcache, position=next_position) # (batch = 1)
            # update position for RoPE
            next_position +=1
            # probabilistically get the next token
            new_token = get_token_from_logits(logits[0][-1], param, temp)
            # add it to the list
            generated_tokens = torch.cat((generated_tokens, new_token), dim = 0)
            # Uncomment this next line to prints the tokens generated one at a time
            # print_sequence(generated_tokens)
            # make sure kvcahce doesn't get longer than context window
            for blockcache in kvcache:
                blockcache['K'] = blockcache['K'][:, :, -context_window:, :]
                blockcache['V'] = blockcache['V'][:, :, -context_window:, :]
    return generated_tokens

  
# Set for evaluation
model.eval()
# randomly checking sequence 13... but in needs a 2D tensor.  torch.unsqueeze converts from (x) to (1,x)
startertokens = inputs[5][:4]
context_window = 100
tokens_to_generate = 10
temperature = 1
topk = 2
topp = 0.9
'''
logits, _ = model(startertokens.unsqueeze(0))
next_logits = logits[0, -1]
print(next_logits)
logits, cache = model(startertokens[:-1].unsqueeze(0))
logits4, _ = model(startertokens[-1:].unsqueeze(0),cache)
next_logits = logits4[0, -1]
print(next_logits)
'''
returned_tokens = generator(startertokens, tokens_to_generate, context_window, model, param=topp, temp=1)

print("Prompt:")
print_sequence(startertokens)
print("Full sequence:")
print_sequence(returned_tokens)

print("\n")
print("--------------------------------------------------------")
print("-"*10, "Lesson 50 - Safetensors & GGUP", "-"*10)
print("--------------------------------------------------------")

# Pretend tensors
tensors = {
    "WQ": torch.randn(8,8),
    "WK": torch.randn(4,8),
    "WV": torch.randn(4,8),
    "bias": torch.randn(8),
}

save_file(tensors, "tiny_model.safetensors")

# My real model tensors
save_file(model.state_dict(), "model.safetensors")
print("state_dict")
for name, tensor in model.state_dict().items():
    print(name, tensor.shape, tensor.dtype)
print("\nsafetensors")
with safe_open("model.safetensors", framework="pt") as f:

    print("Tensors:")

    for key in f.keys():
        tensor = f.get_tensor(key)

        print(
            key,
            "shape =", tuple(tensor.shape),
            "dtype =", tensor.dtype
        )

print("\n")
print("--------------------------------------------------------")
print("-"*10, "Lesson 51 - Quantizers", "-"*10)
print("--------------------------------------------------------")

weights = torch.tensor([
    -1.0,
    -0.7,
    -0.3,
     0.1,
     0.4,
     0.6,
     0.8,
     1.0
])
print(f"Weights: {weights}")

def quantize4(weights):
    bits = 4
    scale = torch.max(torch.abs(weights))/(2**(bits-1)-1)

    quants = torch.round(weights/scale).to(torch.int8)
    return quants, scale

def pack4(quants):
    assert len(quants)%2 == 0, f"quants must have even length"
    new = []
    values = quants+8
    for i in range(0, len(quants)-1, 2):
        low = int(values[i])
        high = int(values[i+1])
        byte = low | (high << 4)
        new.append(byte)
    return torch.tensor(new, dtype=torch.uint8)

def unpack4(packed):
    new = []
    for i in packed:
        byte = int(i)
        new.append((byte & 0x0F) - 8)
        new.append ((byte >> 4) - 8)
    return torch.tensor(new, dtype=torch.int8)

def dequantize4(quants, scale):
    reconstruct = quants * scale
    return reconstruct

quants, scale = quantize4(weights)
print(f"scale: {scale}")
print(f"quantized: {quants}")

packets = pack4(quants)
print(f"packets: {packets}")
unpackets = unpack4(packets)
print(f"unpackets: {unpackets}")

recons = dequantize4(unpackets, scale)
print(f"reconstructed: {recons}")

error = weights - recons
print(f"error: {error}")

print("\n")
print("--------------------------------------------------------")
print("-"*10, "Lesson 52, 53 - Ollama API", "-"*10)
print("--------------------------------------------------------")

# messages is basically the context window for the chat.  It is a list of dicts.
# each dict has two entries:  "role" and "content"
# role can be:
#       "user" - that means me
#       "assistant" - that means the chatbont
#       "system" - not sure what this means... maybe foe settings?
# To keep a conversation going, you have to repeated keep appending to messages with
# alternating roles... user - assistant - user - assistant ... etc
messages = []

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B"
)

while True:
        userinput = input("You: ")
        if userinput == "/bye": break

        # Break into tokens & get token IDs (not necessary, but I want to see it)
        tokens = tokenizer.tokenize(userinput)
        token_ids = tokenizer.encode(userinput)
        print(f"\tTokens: {tokens}")
        print(f"\tToken_ids({len(token_ids)}): {token_ids}")

        # add my prompt to the context
        messages.append({"role":"user", "content":userinput})

        # Now look at what the chat template looks like
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"\tFormatted template: {formatted}")
        print(f"\tFormatted tokens: {tokenizer.tokenize(formatted)}")
        f_tokens_ids = tokenizer.apply_chat_template(messages, tokenize=true, add_generation_prompt=True)
        print(f"\tFormatted Token_ids({len(f_token_ids)}): {f_token_ids}")
        
        # send the context to the chatbot
        response = chat(model="qwen3:0.6b", messages=messages)

        # Response is an object.  to get the text do this:
        answer = response.message.content


        # add the answer to the context, from the "assistant"
        messages.append({"role":"assistant", "content":answer})

        # now print the response
        print("Bot:", answer, "\n")

print("All done")

