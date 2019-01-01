import bcolz
import torch
import pickle
import linecache
import numpy as np 
from torch.utils.data import Dataset, DataLoader

_GLOVE_PATH = '/home/nevronas/word_embeddings/glove_twitter'
_MAX_LEN = 102
_EMB_DIM = 50

def init_glove(glove_path=_GLOVE_PATH): # Run only first time
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='{}/27B.50.dat'.format(glove_path), mode='w')
    with open('{}/glove.twitter.27B.50d.txt'.format(glove_path), 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors.reshape((1193514, _EMB_DIM)), rootdir='{}/27B.50.dat'.format(glove_path), mode='w')
    vectors.flush()
    pickle.dump(words, open('{}/27B.50_words.pkl'.format(glove_path), 'wb'))
    pickle.dump(word2idx, open('{}/27B.50_idx.pkl'.format(glove_path), 'wb'))
    return idx

class OffenseEval(Dataset):
    """OffenseEval dataset."""

    def __init__(self, path, glove_path=_GLOVE_PATH, task='A'):
        self.path = path
        self.glove_path = glove_path
        self.leng = sum(1 for line in open(self.path)) 
        self.glove = self.load_glove()
        self.task = task

    def load_glove(self):
        vectors = bcolz.open('{}/27B.50.dat'.format(self.glove_path))[:]
        words = pickle.load(open('{}/27B.50_words.pkl'.format(self.glove_path), 'rb'))
        word2idx = pickle.load(open('{}/27B.50_idx.pkl'.format(self.glove_path), 'rb'))

        return {w: vectors[word2idx[w]] for w in words}

    def map_index(self, contentstr):
        SUBA, SUBB, SUBC= ['NULL', 'NOT', 'OFF'], ['NULL', 'TIN', 'UNT'], ['NULL', 'IND', 'GRP', 'OTH']
        contents = {}
        contents['instance'] = contentstr[1]
        contents['SUBA'] = SUBA.index(contentstr[2])
        contents['SUBB'] = SUBB.index(contentstr[3])
        contents['SUBC'] = SUBC.index(contentstr[4])
        contents['embeddings'] = np.asarray([self.glove.get(word, self.glove['unk']) for word in  contents["instance"].split(" ")])
        # TODO:  concatenate end of line word embedding, then :
        concat = np.zeros((_MAX_LEN - contents["embeddings"].shape[0], _EMB_DIM))
        contents["embeddings"] = np.concatenate((contents["embeddings"], concat))
        return contents

    def __len__(self):
        return self.leng

    def __getitem__(self, idx):
        if(idx == 0):
            return __getitem__(idx + 1)
        line = linecache.getline(self.path, idx + 1)[:-1]
        contents = line.split("\t")
        contents = self.map_index(contents)
        return contents['embeddings'], contents['SUB{}'.format(self.task)]

if __name__ == '__main__':
    dataset = OffenseEval(path='/home/nevronas/Projects/Personal-Projects/Dhruv/OffensEval/dataset/train-v1/offenseval-training-v1.tsv')
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    dataloader = iter(dataloader)
    for i in range(0, len(dataloader)):
        embeddings, suba = next(dataloader)
        print("{} - {}".format(embeddings.shape, suba.shape))
