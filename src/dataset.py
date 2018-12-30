import bcolz
import torch
import linecache
import numpy as np 
from torch.utils.data import Dataset

_GLOVE_PATH = '/home/nevronas/word_embeddings/glove_twitter'

def init_glove(glove_path=_GLOVE_PATH): # Run only first time
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='{}/6B.50.dat'.format(glove_path), mode='w') # TODO : Install bcloz

    with open('{}/glove.twitter.27B.50d.txt'.format(glove_path), 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir='{}/6B.50.dat'.format(glove_path), mode='w')
    vectors.flush()
    pickle.dump(words, open('{}/6B.50_words.pkl'.format(glove_path), 'wb'))
    pickle.dump(word2idx, open('{}/6B.50_idx.pkl'.format(glove_path), 'wb'))
    return idx

class OffenseEval(Dataset):
    """OffenseEval dataset."""

    def __init__(self, path, glove_path=_GLOVE_PATH):
        self.path = path
        self.glove_path = glove_path
        self.leng = sum(1 for line in open(self.path)) 
        self.glove = self.load_glove()

    def load_glove(self):
        vectors = bcolz.open('{}/6B.50.dat'.format(self.glove_path))[:]
        words = pickle.load(open('{}/6B.50_words.pkl'.format(self.glove_path), 'rb'))
        word2idx = pickle.load(open('{}/6B.50_idx.pkl'.format(self.glove_path), 'rb'))

        return {w: vectors[word2idx[w]] for w in words}

    def map_index(self, contentstr):
        SUBA, SUBB, SUBC= ['NULL', 'NOT', 'OFF'], ['NULL', 'TIN', 'UNT'], ['NULL', 'IND', 'GRP', 'OTH']
        contents['instance'] = contentstr[0]
        contents['SUBA'] = SUBA.index(contentstr[1])
        contents['SUBB'] = SUBB.index(contentstr[2])
        contents['SUBC'] = SUBC.index(contentstr[3])
        contents['embeddings'] = np.asarray([self.glove[word] for word in  contents["instance"].split(" ")[1:]])

        return contents

    def __len__(self):
        return self.leng

    def __getitem__(self, idx):
        line = linecache.getline(self.path, idx + 1)[:-1]
        contents = line.split("\t")
        contents = map_index(contents)
        return contents

if __name__ == '__main__':
    init_glove()