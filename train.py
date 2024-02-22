import argparse
import os
import time
import logging
import datetime
from misc import printf, save_model, MFIDataLoader, cal_loss, AverageMeter, save_args, intersectionAndUnionGPU, progress_bar
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from model import Model
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np
from tensorboardX import SummaryWriter
import glob
import random
import pandas as pd

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if not os.path.exists('Result'):
    os.makedirs('Result')
writer = SummaryWriter('Result')

def main(args):   
    
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    if not os.path.isfile(os.path.join(args.checkpoint, "logger.txt")):
        open(os.path.join(args.checkpoint, "logger.txt"), 'w')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "logger.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)


    printf(f"args: {args}", screen_logger)
    printf('==> Building model..', screen_logger)

    net = Model(args.device, args.inchannel, args.classes, args.feat_scales, args.feature_outchannel, args.num_points, args.time_steps, args.channel_multiplier).to(args.device)
    criterion = cal_loss
    if args.device == 'cuda:0':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

   
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    best_test_acc = 0.  
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_train_mIoU = 0.
    best_test_mIoU = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0 
    optimizer_dict = None

    if not os.path.isfile(os.path.join(args.checkpoint, "best_checkpoint.pth")):
        save_args(args)

    else:
        printf(f"Resuming last checkpoint from {args.checkpoint}", screen_logger)
        checkpoint_path = os.path.join(args.checkpoint, "best_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        optimizer_dict = checkpoint['optimizer']

    printf('==> Preparing data..', screen_logger)
    
    trainfilelst = glob.glob(args.trainfilepath+'\\*.TXT')[:100]
    testfilelst = glob.glob(args.testfilepath+'\\*.TXT')[:100]
    random.shuffle(trainfilelst)
    random.shuffle(testfilelst)
    
    train_loader = DataLoader(MFIDataLoader(trainfilelst, 900, True),
                                batch_size=args.batch_size,
                                pin_memory=True,
                                num_workers=0,
                                shuffle=True, drop_last=True)
    test_loader = DataLoader(MFIDataLoader(testfilelst, 900, True),
                                batch_size=args.batch_size,
                                pin_memory=True,
                                num_workers=0,
                                shuffle=True,drop_last=True)

    optimizer = torch.optim.Adam(net.parameters(),lr=args.learning_rate)   
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr, last_epoch=- 1)


    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, scheduler.get_lr()), screen_logger)
        train_out = train(net, train_loader, optimizer, criterion, args.device, args,
                        intersection_meter,
                        union_meter,
                        target_meter)
        test_out = validate(net, test_loader, criterion, args.device, args,
                        intersection_meter,
                        union_meter,
                        target_meter)
        scheduler.step()

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_train_mIoU = train_out["mIoU"] if (train_out["mIoU"] > best_train_mIoU) else best_train_mIoU
        best_test_mIoU = test_out["mIoU"] if (test_out["mIoU"] > best_test_mIoU) else best_test_mIoU

        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

        save_model(
            net, epoch, path='checkpoints', acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc, 
            best_train_acc=best_train_acc,
            best_test_acc_avg=best_test_acc_avg,
            best_train_acc_avg=best_train_acc_avg,
            best_train_mIoU = best_train_mIoU,
            best_test_mIoU = best_test_mIoU,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict()
        )
        
        writer.add_scalars('train/test_loss', {'train_loss':train_out['loss'],'test_loss':test_out['loss']}, epoch)
        writer.add_scalars('train/test_acc', {'train_acc':train_out['acc'], 'test_acc':test_out['acc']}, epoch)
        writer.add_scalars('train/test_mIoU', {'train_mIoU':train_out['mIoU'], 'test_mIoU':test_out['mIoU']}, epoch)
        
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% mIoU:{train_out['mIoU']} time:{train_out['time']}s", screen_logger)
        printf(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}%  mIoU:{test_out['mIoU']} time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n", screen_logger)
        
        writer.close()
    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2, screen_logger)
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++", screen_logger)
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++", screen_logger)
    printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++", screen_logger)
    printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++", screen_logger)
    printf(f"++  Best Train mIoU: {best_train_mIoU} | Best Test mIoU: {best_test_mIoU}  ++", screen_logger)
    printf(f"++++++++" * 5, screen_logger)


def train(net, trainloader, optimizer, criterion, device, args,
    intersection_meter,
    union_meter,
    target_meter):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1) 
        optimizer.zero_grad()
        logits = net(data.permute(0,2,1))
        logits = logits.view(-1, args.classes)
        label = label.to(torch.int64).view(-1, 1)[:,0]
        loss = criterion(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        intersection, union, target = intersectionAndUnionGPU(preds, label, args.classes)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | mIoU: %.3f |Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), np.mean(intersection_meter.sum / (union_meter.sum + 1e-10)),100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "mIoU": mIoU,
        "time": time_cost
    }


def validate(net, testloader, criterion, device, args, 
    intersection_meter,
    union_meter,
    target_meter):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data.permute(0,2,1))
            logits = logits.view(-1,args.classes)
            label = label.to(torch.int64).view(-1, 1)[:, 0]
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            
            intersection, union, target = intersectionAndUnionGPU(preds, label, args.classes)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | mIoU: %.3f |Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), np.mean(intersection_meter.sum / (union_meter.sum + 1e-10)),100. * correct / total, correct, total))


    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "mIoU": mIoU,
        "time": time_cost
    }

def pc_normalize(pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


if __name__ == '__main__':
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint',default="checkpoints")
    parser.add_argument('--trainfilepath', type=str, default='C:\\Users\Administrator\\Desktop\\pointMLP-pytorch-main\\点云文件\\train',\
                        help='Trainfile path')
    parser.add_argument('--testfilepath', type=str, default='C:\\Users\Administrator\\Desktop\\pointMLP-pytorch-main\\点云文件\\test',\
                        help='Trainfile path')
    parser.add_argument('--feat_scales', type=list, default=[0,1,2,3], help='Hyperparameters')
    parser.add_argument('--feature_kernels', type=list, default=[32,32,32,32], help='Hyperparameters')
    parser.add_argument('--feature_outchannel', type=list, default=[16,32,64,128], help='Hyperparameters')
    parser.add_argument('--time_steps', type=list, default=[0,0,0,0], help='Hyperparameters')
    parser.add_argument('--channel_multiplier', type=list, default=[1,2,4,8], help='Hyperparameters')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default="MFI", help='model name')
    parser.add_argument('--epoch', default=10240, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=900, help='Point Number')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--min_lr', default=1e-7, type=float, help='min lr')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--inchannel', default=6, type=int, help='feature dimension')
    parser.add_argument('--classes', default=7, type=int, help='class number')
    args = parser.parse_args()
    main(args)
