from torchtext.datasets import WikiText2
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import nltk
import string
import pickle

#train = WikiText2(root = '.data', split = ('train', 'valid', 'test'))[0]
#translator = str.maketrans('','',string.punctuation)
#sentences = list(filter(lambda x: len(x) > 3, [sentence.translate(translator).lower().split() for sentence in list(train)]))
sentences = ['''Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks.'''.lower().split()]

all_words = [word for sentence in sentences for word in sentence]

# generate training skipgrams
skipgrams = []

for sentence in sentences:
    skipgrams += list(nltk.skipgrams(sentence, 2, 1))

with open('skipgrams.pkl', 'wb') as f:
        pickle.dump(skipgrams, f)

# get full vocabulary of corpus
# words are associated with integer between 1 and the 
# length of the vocabulary
vocab = build_vocab_from_iterator(sentences)

with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

