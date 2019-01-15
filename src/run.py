import os
import gc
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import *
from dataset import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch OffenseEval - run dataset.py first for word embeddings')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate') # NOTE :  change for diff models
parser.add_argument('--batch_size', default=25, type=int)
parser.add_argument('--resume', '-r', type=int, default=0, help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--subtask', default='C', help="Sub-task for OffensEval")
parser.add_argument('--embedding_length', default=50, type=int)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, tsepoch, tstep = 0, 0, 0

criterion = torch.nn.CrossEntropyLoss()

print('==> Preparing data..')

'''def collate_fn(data):
    data = list(filter(lambda x: type(x[1]) != int, data))
    audios, captions = zip(*data)
    data = None
    del data
    audios = torch.stack(audios, 0)
    return audios, captions'''

classes = {"A" : 2, "B" : 2, "C" : 3}
print('==> Creating network..')
net = AttentionModel(args.batch_size, [2, 2, 3], 25, args.embedding_length)
net = net.to(device)

if(args.resume):
    if(os.path.isfile('../save/network.ckpt')):
        net.load_state_dict(torch.load('../save/network.ckpt'))
        print('==> Network : loaded')

    if(os.path.isfile("../save/info.txt")):
        with open("../save/info.txt", "r") as f:
            tsepoch, tstep = (int(i) for i in str(f.read()).split(" "))
        print("=> Network : prev epoch found")
else :
    with open("../save/logs/train_loss.log", "w+") as f:
        pass 


def train_network(epoch):
    global tstep
    print('\n=> Epoch: {}'.format(epoch))
    net.train()
    
    dataset = OffenseEval(path='/home/nevronas/Projects/Personal-Projects/Dhruv/OffensEval/dataset/train-v1/offenseval-training-v1.tsv')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader = iter(dataloader)

    train_loss, accu1, accu2, accu3 = 0.0, 0.0, 0.0, 0.0
    le = len(dataloader) - 1
    params = net.parameters()     
    optimizer = torch.optim.Adam(params, lr=args.lr) 

    for i in range(tstep, le):
        contents = next(dataloader)
        inputs = contents[0].type(torch.FloatTensor).to(device)
        target_a, target_b, target_c = [contents[i].type(torch.LongTensor).to(device) for i in range(1, len(contents), 1)]
        targets = [target_a, target_b, target_c]
        optimizer.zero_grad()
        y_preds = net(inputs)
        preds = [torch.max(y_pred, 1)[0].type(torch.LongTensor) for y_pred in y_preds]
        
        loss = criterion(y_preds[0], target_a) + criterion(y_preds[1], target_b) + criterion(y_preds[2], target_c)
        tl = loss.item()
        loss.backward()
        optimizer.step()

        acc1 = f1_score(target_a.detach().cpu().numpy(), preds[0].detach().cpu().numpy(), average='macro')
        acc2 = f1_score(target_b.detach().cpu().numpy(), preds[1].detach().cpu().numpy(), average='macro')
        acc3 = f1_score(target_c.detach().cpu().numpy(), preds[2].detach().cpu().numpy(), average='macro')
        
        train_loss += tl
        accu1 += acc1 
        accu2 += acc2 
        accu3 += acc3

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(net.state_dict(), '../save/network.ckpt')
        with open("../save/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        with open("../save/logs/train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(tl))

        progress_bar(i, len(dataloader), 'Loss: {}, F1s: {} - {} - {}'.format(tl, accu1, accu2, accu3))

    tstep = 0
    del dataloader
    print('=> Network : Epoch [{}/{}], Loss:{:.4f}, F1:{:.4f} - {:.4f} - {:.4f}'.format(epoch + 1, args.epochs, train_loss / le, accu1 / le, accu2 / le, accu3 / le))


def test():
    # TODO
    pass


for epoch in range(tsepoch, tsepoch + args.epochs):
    train_network(epoch)

#test()