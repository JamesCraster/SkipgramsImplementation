from torch import nn
import torch
import pickle
import random
import matplotlib.pyplot as plt
import sklearn.decomposition

with open('skipgrams.pkl', 'rb') as f:
    skipgrams = pickle.load(f)

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

print(skipgrams)
print(vocab)

# convert word to integer index in vocab
# convert from integer index in vocab to one-hot encoding
def one_hot(vocab, keys):
    if isinstance(keys, str):
        keys = [keys]
    return torch.nn.functional.one_hot(torch.tensor(vocab(keys)), num_classes=len(vocab))

device = "mps"
vocabulary_size = len(vocab)
embedding_dimension = 20

class SkipgramNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        first = nn.Linear(vocabulary_size, embedding_dimension, bias=False)
        nn.init.normal_(first.weight)
        second = nn.Linear(embedding_dimension, vocabulary_size, bias=False)
        nn.init.normal_(second.weight)

        self.stack = nn.Sequential(
            first,
            second
            #nn.Softmax()
        )
        
    
    def forward(self, input):
        output = self.stack(input)
        return output
 
model = SkipgramNetwork().to(device)

# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# make my own on-hot vector (a bit clumsy)

def list_to_tensor(x):
    return torch.tensor(x, dtype=torch.float, device=device)


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.08)

# for x in skipgrams:
#     print(x[0], x[1], list_to_tensor(one_hot(vocab, x[0])))

dataset = []
for word, label in skipgrams:
    dataset.append((list_to_tensor(one_hot(vocab, word)), list_to_tensor(one_hot(vocab, label))))

losses = []

print(len(dataset))

def train_one_epoch(epoch_index):
    optimizer.zero_grad()
    summation = 0
    for word, label in dataset:
        outputs = model(word)
        loss = loss_fn(outputs, label)
        summation += loss.item()
        loss.backward()
    # skipgrams correct both ways around
    for label, word in dataset:
        outputs = model(word)
        loss = loss_fn(outputs, label)
        summation += loss.item()
        loss.backward()
    optimizer.step()
    print(summation/len(dataset))
    losses.append(summation/len(dataset))

# 50 seems work well
epoch_count = 40
for epoch in range(0,epoch_count):
    model.train(True)
    train_one_epoch(epoch)

print(losses)
plt.plot(losses)
plt.show()

embeddings = next(model.parameters())

with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)


def word_to_encoding(word):
    index = vocab.lookup_indices([word])[0]
    return embeddings[:, index]

cos = nn.CosineSimilarity(dim=0)

pca_model = sklearn.decomposition.PCA(3)
print(embeddings.shape)
pca_model.fit(embeddings.cpu().detach().numpy().T)
transformed = pca_model.transform(embeddings.cpu().detach().numpy().T)



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(transformed[:,0],transformed[:,1],transformed[:,2])

for i in range(0,60):
    ax.text(transformed[i,0],transformed[i,1],transformed[i,2], vocab.lookup_token(i))

plt.show()

detached_embeddings = embeddings.cpu().detach().numpy().T


