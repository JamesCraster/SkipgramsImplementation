import pickle
from torch import nn

with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

def word_to_encoding(word):
    index = vocab.lookup_indices([word])[0]
    return embeddings[:, index]

cos = nn.CosineSimilarity(dim=0)

# v = "lennon"
# search_term = word_to_encoding(v)
# dist = 0
# path_of_closeness = []

# for i in range(1,len(vocab)):
#     if(cos(search_term, embeddings[:, i]) > dist):
#         if(vocab.lookup_token(i) != v):
#             path_of_closeness.append(i)
#             print(i)
#             print(vocab.lookup_token(i))
#             dist = cos(search_term, embeddings[:, i])

# print(path_of_closeness)
# print(cos(word_to_encoding("horse"), word_to_encoding("ridden")))
