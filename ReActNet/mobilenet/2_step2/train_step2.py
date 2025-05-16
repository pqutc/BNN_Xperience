import os
import sys
import shutil
import torch
import logging
import argparse
import torch.nn as nn
import time
import torch.utils
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path
import mlflow
import mlflow.pytorch


sys.path.append("../../")

from reactnet import reactnet
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
mlflow.set_tracking_uri("../mlruns")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def save_checkpoint(state, is_best, save):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# DataLoader qui pourrait être utilisé ultérieurement
class NImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
                
        # Trouver toutes les classes et fichiers
        self.samples = []
        self.class_to_idx = {}
                
        for idx, class_dir in enumerate(sorted(Path(data_path).iterdir())):
            if not class_dir.is_dir():
                continue
                        
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
                    
            for npz_file in class_dir.glob("*.npz"):
                self.samples.append((str(npz_file), idx))
                
        print(f"Chargé {len(self.samples)} échantillons de {len(self.class_to_idx)} classes")
            
    def __len__(self):
        return len(self.samples)
            
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.load(file_path)
        image = data['image'].astype(np.float32)
                
        return image, label

parser = argparse.ArgumentParser("birealnet18")
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', type=int, default=256, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--teacher', type=str, default='resnet34', help='path of ImageNet')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()

CLASSES = 100

if not os.path.exists('log'):
    os.mkdir('log')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('log/log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()
    mlflow.set_experiment("reactnet_training_step2")
    with mlflow.start_run(run_name="reactnet_step2") as run:
        run_id = run.info.run_id
        logging.info(f"MLflow Run ID: {run_id}")
        # Enregistrement des hyperparamètres
        mlflow.log_params({
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "label_smooth": args.label_smooth,
            "teacher": args.teacher
        })
        cudnn.benchmark = True
        cudnn.enabled=True
        logging.info("args = %s", args)

        model_student = reactnet(num_classes=CLASSES)
        logging.info('student:')
        logging.info(model_student)
        model_student = nn.DataParallel(model_student).cuda()

        criterion = nn.CrossEntropyLoss().cuda()
        criterion_kd = nn.CrossEntropyLoss()

        all_parameters = model_student.parameters()
        weight_parameters = []
        for pname, p in model_student.named_parameters():
            if p.ndimension() == 4 or 'conv' in pname:
                weight_parameters.append(p)
        weight_parameters_id = list(map(id, weight_parameters))
        other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

        optimizer = torch.optim.Adam(
                [{'params' : other_parameters},
                {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
                lr=args.learning_rate,)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
        start_epoch = 0
        best_top1_acc= 0

        checkpoint_tar = os.path.join(args.save, 'checkpoint_ba.pth.tar')
        checkpoint = torch.load(checkpoint_tar)
        model_student.load_state_dict(checkpoint['state_dict'], strict=False)

        checkpoint_tar = os.path.join(args.save, 'checkpoint.pth.tar')
        if os.path.exists(checkpoint_tar):
            logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
            checkpoint = torch.load(checkpoint_tar)
            start_epoch = checkpoint['epoch'] + 1
            best_top1_acc = checkpoint['best_top1_acc']
            model_student.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))

        # adjust the learning rate according to the checkpoint
        for epoch in range(start_epoch):
            scheduler.step()

        # load training data
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'test')

                

        train_dataset = NImageNetDataset(traindir)
        test_dataset = NImageNetDataset(valdir)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )




        # train the model
        epoch = start_epoch
        while epoch < args.epochs:
            train_obj, train_top1_acc,  train_top5_acc = train(epoch,  train_loader, model_student, None, criterion_kd, optimizer, scheduler)
            valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model_student, criterion, args)

            mlflow.log_metric("train_loss", train_obj, step=epoch)
            mlflow.log_metric("train_acc", train_top1_acc, step=epoch)
            mlflow.log_metric("val_loss", valid_obj, step=epoch)
            mlflow.log_metric("val_acc", valid_top1_acc, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
            mlflow.log_metric("train_top5_acc", train_top5_acc, step=epoch)
            mlflow.log_metric("val_top5_acc", valid_top5_acc, step=epoch)

            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                is_best = True

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model_student.state_dict(),
                'best_top1_acc': best_top1_acc,
                'optimizer' : optimizer.state_dict(),
                }, is_best, args.save)

            epoch += 1

        training_time = (time.time() - start_t) / 3600
        print('total training time = {} hours'.format(training_time))


def train(epoch, train_loader, model_student, model_teacher, criterion, optimizer, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')


    model_student.train()
    end = time.time()
    scheduler.step()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits_student = model_student(images)
        loss = criterion(logits_student, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


    return losses.avg, top1.avg, top5.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
