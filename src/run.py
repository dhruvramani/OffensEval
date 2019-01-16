import os
import gc
import csv
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
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--resume', '-r', type=int, default=0, help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--subtask', default='C', help="Sub-task for OffensEval")
parser.add_argument('--embedding_length', default=50, type=int)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, tsepoch, tstep = 0., 0, 0#[0., 0., 0.], 0, 0

criterion = torch.nn.CrossEntropyLoss(reduction='none')

print('==> Preparing data..')

'''def collate_fn(data):
    data = list(filter(lambda x: type(x[1]) != int, data))
    audios, captions = zip(*data)
    data = None
    del data
    audios = torch.stack(audios, 0)
    return audios, captions'''

classes = {"A" : 3, "B" : 3, "C" : 4}
print('==> Creating network..')
net = AttentionModel(args.batch_size, [3, 3, 4], 25, args.embedding_length)
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
    global best_acc
    print('\n=> Epoch: {}'.format(epoch))
    net.train()
    
    dataset = OffenseEval(path='/home/nevronas/Projects/Personal-Projects/Dhruv/OffensEval/dataset/train-v1/offenseval-training-v1.tsv')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) #, collate_fn=collate_fn)
    dataloader = iter(dataloader)

    train_loss, accu1, accu2, accu3 = 0.0, 0.0, 0.0, 0.0
    le = len(dataloader) - 1
    params = net.parameters()     
    optimizer = torch.optim.Adam(params, lr=args.lr) 

    # Training for task A
    for i in range(tstep, le):
        contents = next(dataloader)
        inputs = contents[0].type(torch.FloatTensor).to(device)
        target_a = contents[1].type(torch.LongTensor).to(device) #, target_b, target_c = [contents[i].type(torch.LongTensor).to(device) for i in range(1, len(contents), 1)]
        #targets = [target_a, target_b, target_c]
        
        suban = target_a.detach().cpu().numpy() #, subbn, subcn = target_a.detach().cpu().numpy(), target_b.detach().cpu().numpy(), target_c.detach().cpu().numpy()
        mask1 = np.where(suban == 0) #, mask2, mask3 = np.where(suban == 0), np.where(subbn == 0), np.where(subcn == 0)

        optimizer.zero_grad()
        y_preds = net(inputs)
        #preds = [torch.max(y_pred, 1)[0].type(torch.LongTensor) for y_pred in y_preds]
        
        l1o = criterion(y_preds, target_a) #, l2o, l3o = criterion(y_preds[0], target_a), criterion(y_preds[1], target_b), criterion(y_preds[2], target_c)
        l1 = l1o.detach().cpu().numpy() #, l2, l3 = l1o.detach().cpu().numpy(), l2o.detach().cpu().numpy(), l3o.detach().cpu().numpy()
        l1[mask1] = 0 #, l2[mask2], l3[mask3] = 0, 0, 0
        l1 = torch.Tensor(l1).to(device) #, l2, l3 = torch.Tensor(l1).to(device), torch.Tensor(l2).to(device), torch.Tensor(l3).to(device)

        loss = torch.mean((l1 * l1o) / 2) #+ torch.mean((l2 * l2o) / 2) + torch.mean((l3 * l3o) / 2)
        tl = loss.item()
        loss.backward()
        optimizer.step()

        acc1 = f1_score(target_a.detach().cpu().numpy(), torch.max(y_preds, 1)[0].type(torch.LongTensor).detach().cpu().numpy(), average='macro')
        #acc2 = f1_score(target_b.detach().cpu().numpy(), preds[1].detach().cpu().numpy(), average='macro')
        #acc3 = f1_score(target_c.detach().cpu().numpy(), preds[2].detach().cpu().numpy(), average='macro')
        
        train_loss += tl
        accu1 += acc1 
        #accu2 += acc2 
        #accu3 += acc3

        gc.collect()
        torch.cuda.empty_cache()

        torch.save(net.state_dict(), '../save/network.ckpt')
        with open("../save/info.txt", "w+") as f:
            f.write("{} {}".format(epoch, i))

        with open("../save/logs/train_loss.log", "a+") as lfile:
            lfile.write("{}\n".format(tl))

        progress_bar(i, len(dataloader), 'Loss: {}, F1s: {} '.format(tl, acc1))#, acc2, acc3)) # "{} - {}"

    tstep = 0
    del dataloader
    print('=> Network : Epoch [{}/{}], Loss:{:.4f}, F1:{:.4f}'.format(epoch + 1, args.epochs, train_loss / le, accu1 / le)) #,  accu2 / le, accu3 / le))
    #accu = [accu1/le, accu2/le, accu3/le]
    #best_acc = [max(best_acc[i], accu[i]) for i in range(3)]
    old_best = best_acc
    best_acc = max(best_acc, accu1/le)
    if(best_acc != old_best):
        torch.save(net.state_dict(), '../save/best.ckpt')
    print("Best Metrics : {}".format(best_acc))

def test():
    global net
    net.load_state_dict(torch.load('../save/success/1.ckpt'))
    
    dataset = OffenseEval(path='/home/nevronas/Projects/Personal-Projects/Dhruv/OffensEval/dataset/testset-taska.tsv', train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size) #, collate_fn=collate_fn)
    dataloader = iter(dataloader)
    test_dict = ['NOT', 'OFF']

    with open('../save/test.tsv', 'w+') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(['id', 'tweet', 'subtask_a'])

        for i in range(0, len(dataloader) - 1):
            contents = next(dataloader)
            inputs = contents[2].type(torch.FloatTensor).to(device)
            y_preds = net(inputs)
            clas = torch.max(y_preds, 1)[0].type(torch.LongTensor) - 1
            clas = clas.tolist()

            for i in range(len(clas)):
                if(int(clas[i]) > 1):
                    clas[i] = 1
                print(test_dict[int(clas[i])])
                tsv_writer.writerow([contents[0][i], contents[1][i], test_dict[int(clas[i])] ])

#for epoch in range(tsepoch, tsepoch + args.epochs):
#    train_network(epoch)

test()