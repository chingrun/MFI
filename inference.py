import argparse
import os
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from misc import pc_normalize
from model import Model
import numpy as np
import glob
import threading
import sys
import pandas as pd

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def infer(args):
    start = time.time()  
    filelist = glob.glob(args.inferfilepath+'\\*.TXT')  
    alllength = len(filelist)
    loct=10
    patchsize = args.num_points
    ind = 0    
    model = Model(args.device, args.inchannel,args.classes, args.feat_scales, args.feature_outchannel, args.num_points, args.time_steps, args.channel_multiplier).to(args.device)
    model.load_state_dict(torch.load(args.modelpath)['net'])
    model.eval()
    alllength = len(filelist)
    ind = 0

    evenelements = int(alllength//10)
    filelist1 = filelist[:evenelements]
    filelist2 = filelist[evenelements:2*evenelements]
    filelist3 = filelist[2*evenelements:3*evenelements]
    filelist4 = filelist[3*evenelements:4*evenelements]
    filelist5 = filelist[4*evenelements:5*evenelements]
    filelist6 = filelist[5*evenelements:6*evenelements]
    filelist7 = filelist[6*evenelements:7*evenelements]
    filelist8 = filelist[7*evenelements:8*evenelements]
    filelist9 = filelist[8*evenelements:9*evenelements]
    filelist10 = filelist[9*evenelements:]


    t1 = MyThread(patchsize, filelist1, model, loct, evenelements, ind)
    t2 = MyThread(patchsize, filelist2, model, loct, evenelements, ind)
    t3 = MyThread(patchsize, filelist3, model, loct, evenelements, ind)
    t4 = MyThread(patchsize, filelist4, model, loct, evenelements, ind)
    t5 = MyThread(patchsize, filelist5, model, loct, evenelements, ind)
    t6 = MyThread(patchsize, filelist6, model, loct, evenelements, ind)
    t7 = MyThread(patchsize, filelist7, model, loct, evenelements, ind)
    t8 = MyThread(patchsize, filelist8, model, loct, evenelements, ind)
    t9 = MyThread(patchsize, filelist9, model, loct, evenelements, ind)
    t10 = MyThread(patchsize, filelist10, model, loct, evenelements, ind)

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()
    t9.start()
    t10.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()
    t7.join()
    t8.join()
    t9.join()
    t10.join()

    t1_pdscaninteg = t1.pdscaninteg
    t2_pdscaninteg = t2.pdscaninteg
    t3_pdscaninteg = t3.pdscaninteg
    t4_pdscaninteg = t4.pdscaninteg
    t5_pdscaninteg = t5.pdscaninteg
    t6_pdscaninteg = t6.pdscaninteg
    t7_pdscaninteg = t7.pdscaninteg
    t8_pdscaninteg = t8.pdscaninteg
    t9_pdscaninteg = t9.pdscaninteg
    t10_pdscaninteg = t10.pdscaninteg

    pdsave = pd.concat([t1_pdscaninteg,t2_pdscaninteg, t3_pdscaninteg,t4_pdscaninteg, t5_pdscaninteg,t6_pdscaninteg,t7_pdscaninteg, t8_pdscaninteg,t9_pdscaninteg, t10_pdscaninteg],axis=0)
    pdsave.to_csv(args.savefile, sep=',', index=0,header=None)
    end = time.time()
    print('Data classified in: {}'.format(end - start))

class MyThread(threading.Thread):
    def __init__(self, patchsize, filelist, model, loct, alllength, ind) -> None:
        super().__init__()
        self.patchsize = patchsize
        self.filelist = filelist
        self.loct = loct
        self.alllength = alllength
        self.ind = ind
        self.model = model

    def run(self):
        self.pdscaninteg = pd.DataFrame(data=None)
        self.ftoutinteg = pd.DataFrame(data=None)         
        
        for id in range(0,len(self.filelist)):
            pdscan = pd.read_csv(self.filelist[id], sep=',', header=None)
            pdscan[self.loct-1]=pdscan[self.loct-1].astype('Int64')
            pdscan.insert(loc=self.loct, column=self.loct, value=32767)

           

            inferfeature = np.loadtxt(self.filelist[id], delimiter=',').astype(np.float32)
            point_setf = np.zeros((self.patchsize, self.loct-4))
            point_setf[:, 0:3] = pc_normalize(inferfeature[:self.patchsize, 0:3])
            point_setf[:, 3:5] = inferfeature[:self.patchsize, 3:5]
            point_setf[:, 5] = inferfeature[:self.patchsize, -2]
            point_setf = torch.from_numpy(point_setf).type(torch.Tensor).unsqueeze(0)
        
            y_pred = self.model(point_setf.to(args.device))
            pred_choice = y_pred.data.max(2)[1][0]
            pdscan[self.loct]=np.array(np.ones(len(pdscan[self.loct]))*pred_choice.cpu().numpy(),dtype='Int64')

            self.ind+=1
            sys.stdout.write('\rProgress ' + str("{0:.2f}".format(round(self.ind / (self.alllength)*100, 2))) + "%")
            self.pdscaninteg = pd.concat([self.pdscaninteg, pdscan], axis=0)
                
            del pdscan
        return self.pdscaninteg

if __name__ == '__main__':
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint',default="checkpoints")
    parser.add_argument('--modelpath', type=str,\
                        help='Checkpoint path')
    parser.add_argument('--inferfilepath', type=str,\
                        help='TXT inferfile path')
    parser.add_argument('--savefile', type=str,\
                        help='Savefile path')
    parser.add_argument('--feat_scales', type=list, default=[0,1,2,3], help='Hyperparameters')
    parser.add_argument('--feature_kernels', type=list, default=[32,32,32,32], help='Hyperparameters')
    parser.add_argument('--feature_outchannel', type=list, default=[16,32,64,128], help='Hyperparameters')
    parser.add_argument('--time_steps', type=list, default=[0,0,0,0], help='Hyperparameters')
    parser.add_argument('--channel_multiplier', type=list, default=[1,2,4,8], help='Hyperparameters')
    parser.add_argument('--model', default="MFI", help='model name')
    parser.add_argument('--num_points', type=int, default=900, help='Point Number')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--min_lr', default=1e-7, type=float, help='min lr')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--inchannel', default=6, type=int, help='feature dimension')
    parser.add_argument('--classes', default=7, type=int, help='class number')
    args = parser.parse_args()
    infer(args)