import os
import torch
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.cuda import amp
from scipy.io import savemat
import torch.nn.functional as F
import tonic.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.nn.modules import loss
import pandas as pd
import torchvision.models as models

os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'


Begin_epoch = 0
Max_epoch = 64
Learning_rate = 1e-3
Weight_decay = 0
Momentum = 0.9
Top_k = 5
AMP = False
REPRESENTATION_CHOICE = "_2_presence"



Name_suffix = '_step1'
Test_every_iteration = None
Savemodel_path = './savemodels/'
Record_path = './recorddata/'
if not os.path.exists(Savemodel_path):
    os.mkdir(Savemodel_path)
if not os.path.exists(Record_path):
    os.mkdir(Record_path)

#Dataset_path = '../data/N-imagenet_preprocessed'+REPRESENTATION_CHOICE+'/'
Dataset_path = '../data/N-imagenet_preprocessed'+'_2_presence_small'#+REPRESENTATION_CHOICE+'/'
train_path = os.path.join(Dataset_path, 'train')
test_path = os.path.join(Dataset_path, 'test')
Batch_size = 128
Workers = 8
Targetnum = 100
# Création des répertoires nécessaires
os.makedirs(Savemodel_path, exist_ok=True)
os.makedirs(Record_path, exist_ok=True)
os.makedirs(Dataset_path, exist_ok=True)

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
        image = data['image']
        
        return image, label
        

train_dataset = NImageNetDataset(train_path)
test_dataset = NImageNetDataset(test_path)

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=Batch_size, shuffle=True, num_workers=Workers, pin_memory=True, drop_last=True
)
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=Batch_size, shuffle=False, num_workers=Workers, pin_memory=True, drop_last=False
)


def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)


def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-3, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grads)






net = models.resnet50(weights=None)
# Modifier la première couche pour accepter 2 canaux au lieu de 3
net.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Charger les poids préentraînés pour toutes les autres couches
pretrained_state_dict = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).state_dict()
model_state_dict = net.state_dict()

# Ne pas charger les poids de la première couche conv1
for name, param in pretrained_state_dict.items():
    if name not in ['conv1.weight']:
        model_state_dict[name].copy_(param)

# Modifier la dernière couche pour correspondre à votre nombre de classes
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 100)
net = nn.DataParallel(net).cuda()





max_test_acc = 0.
if Begin_epoch!=0:
    net.load_state_dict(torch.load(Savemodel_path + f'epoch{Begin_epoch-1}{Name_suffix}.h5'))
    max_test_acc = np.load(Savemodel_path + f'max_acc{Name_suffix}.npy')
    max_test_acc = max_test_acc.item()

scaler = amp.GradScaler() if AMP else None
Test_top1 = []
Test_topk = []
Test_lossall = []
Epoch_list = []
Iteration_list = []
Train_lossall = []
Train_top1 = []
Train_topk = []

all_parameters = net.parameters()
weight_parameters = []
for pname, p in net.named_parameters():
    if (p.ndimension() == 4 or 'conv' in pname) and 'L_alpha' not in pname:
        weight_parameters.append(p)
weight_parameters_id = list(map(id, weight_parameters))
other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

criterion_train = nn.CrossEntropyLoss()
criterion_test = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params' : other_parameters, 'weight_decay' : 0., 'initial_lr': Learning_rate},
    {'params' : weight_parameters, 'weight_decay' : Weight_decay, 'initial_lr': Learning_rate}],
    lr = Learning_rate)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/Max_epoch), last_epoch=Begin_epoch-1)


def test_model(net, max_test_acc, data_loader=test_data_loader, criterion=criterion_test, epoch=None, iteration=None, record=True):
    net.eval()
    test_samples = 0
    test_loss = 0
    test_acc_top1 = 0
    test_acc_topk = 0
    
    with torch.no_grad():
        for img, label in tqdm(data_loader):
            img = img.cuda().float()
            label = label.cuda()
            label_onehot = F.one_hot(label, Targetnum).float()
            
            out_fr = net(img)
            loss = criterion(out_fr, label)
                
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()

            test_acc_top1 += (out_fr.argmax(1) == label).float().sum().item()
            _, pred = out_fr.topk(Top_k, 1, True, True)
            test_acc_topk += torch.eq(pred, label.view(-1,1)).float().sum().item()
    
    test_loss /= test_samples
    test_acc_top1 /= test_samples
    test_acc_topk /= test_samples

    if test_acc_top1 >= max_test_acc:
        max_test_acc = test_acc_top1
        torch.save(net.state_dict(), Savemodel_path + f'max_acc{Name_suffix}.h5')
        np.save(Savemodel_path + f'max_acc{Name_suffix}.npy', np.array(max_test_acc))

    if record:
        assert epoch is not None, "epoch is None!"
        assert iteration is not None, "iteration is None!"
        
        Epoch_list.append(epoch+1)
        Iteration_list.append(iteration+1)
        Test_top1.append(test_acc_top1)
        Test_topk.append(test_acc_topk)
        Test_lossall.append(test_loss)

        record_data = np.array([Epoch_list, Iteration_list, Test_top1, Test_topk, Test_lossall]).T
        mdic = {f'Record_data':record_data, f'Record_meaning':['Epoch_list', 'Iteration_list', 'Test_top1', f'Test_top{Top_k}', 'Test_loss']}

        savemat(Record_path + f'Test_{Begin_epoch}_{epoch}{Name_suffix}.mat',mdic)
        if os.path.exists(Record_path + f'Test_{Begin_epoch}_{epoch-1}{Name_suffix}.mat'):
            os.remove(Record_path + f'Test_{Begin_epoch}_{epoch-1}{Name_suffix}.mat')

    return test_loss, test_acc_top1, test_acc_topk, max_test_acc

def train_model(net, max_test_acc, epoch, data_loader=train_data_loader, optimizer=optimizer, criterion=criterion_train, scaler=scaler, record=True):
    train_samples = 0
    train_loss = 0
    train_acc_top1 = 0
    train_acc_topk = 0
    
    for i, (img, label) in enumerate(tqdm(data_loader)):
        net.train()
        img = img.cuda().float()
        label = label.cuda()
        label_onehot = F.one_hot(label, Targetnum).float()
        
        if AMP:
            with amp.autocast():
                out_fr = net(img)
                loss = criterion(out_fr, out_teacher)
        else:
            out_fr = net(img)
            out_teacher = label#model_teacher(img)
            loss = criterion(out_fr, out_teacher)
            
        train_samples += label.numel()
        train_loss += loss.item() * label.numel()

        train_acc_top1 += (out_fr.argmax(1) == label).float().sum().item()
        _, pred = out_fr.topk(Top_k, 1, True, True)
        train_acc_topk += torch.eq(pred, label.view(-1,1)).float().sum().item()
        
        optimizer.zero_grad()
        if AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            parameters_list = []
            for name, p in net.named_parameters():
                if not 'fc' in name:
                    parameters_list.append(p)
            adaptive_clip_grad(parameters_list, clip_factor=0.02)
            
            optimizer.step()

        if Test_every_iteration is not None:
            if (i+1) % Test_every_iteration == 0:
                test_loss, test_acc_top1, test_acc_topk, max_test_acc = test_model(net, max_test_acc, epoch=epoch, iteration=i, record=record)
                print(f'Test_loss: {test_loss:.4f}, Test_acc_top1: {test_acc_top1:.4f}, Test_acc_top{Top_k}: {test_acc_topk:.4f}, Max_test_acc: {max_test_acc:.4f}')
    
    train_loss /= train_samples
    train_acc_top1 /= train_samples
    train_acc_topk /= train_samples

    test_loss, test_acc_top1, test_acc_topk, max_test_acc = test_model(net, max_test_acc, epoch=epoch, iteration=i, record=record)

    return train_loss, train_acc_top1, train_acc_topk, test_loss, test_acc_top1, test_acc_topk, max_test_acc




for epoch in range(Begin_epoch, Max_epoch):

    start_time = time.time()
    train_loss, train_acc_top1, train_acc_topk, test_loss, test_acc_top1, test_acc_topk, max_test_acc = train_model(net, max_test_acc, epoch)
    
    record_data = np.array([train_loss, train_acc_top1, train_acc_topk]).T
    mdic = {f'Record_data':record_data, f'Record_meaning':['train_loss', 'train_acc_top1', f'Train_top{Top_k}']}
    savemat(Record_path + f'Train_{Begin_epoch}_{epoch}{Name_suffix}.mat',mdic)
    if os.path.exists(Record_path + f'Train_{Begin_epoch}_{epoch-1}{Name_suffix}.mat'):
        os.remove(Record_path + f'Train_{Begin_epoch}_{epoch-1}{Name_suffix}.mat')

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        
    lr_scheduler.step()

    print(f'''epoch={epoch}, train_acc_top1={train_acc_top1:.4f}, train_acc_top{Top_k}={train_acc_topk:.4f}, train_loss={train_loss:.4f}, test_top1={test_acc_top1:.4f}, test_top{Top_k}={test_acc_topk:.4f}, test_loss={test_loss:.4f}, max_test_acc={max_test_acc:.4f}, total_time={(time.time() - start_time):.4f}, LR={lr:.8f}''')
    
    torch.save(net.state_dict(), Savemodel_path + f'epoch{epoch}{Name_suffix}.h5')
    if os.path.exists(Savemodel_path + f'epoch{epoch-1}{Name_suffix}.h5'):
        os.remove(Savemodel_path + f'epoch{epoch-1}{Name_suffix}.h5')




Confusion_Matrix = torch.zeros((Targetnum, Targetnum))
net.eval()
with torch.no_grad():
    for img, label in tqdm(test_data_loader):
        img = img.cuda().float()
        label = label.cuda()
        out_fr = net(img)
        guess = out_fr.argmax(1)
        for j in range(len(label)):
            Confusion_Matrix[label[j],guess[j]] += 1
acc = Confusion_Matrix.diag()
acc = acc.sum()/Confusion_Matrix.sum()


try:
    torch.save(Confusion_Matrix.cpu(), Record_path+"confusion_matrix.pt")
    pd.DataFrame(Confusion_Matrix.cpu().numpy()).to_csv(Record_path+"confusion_matrix.csv", index=False)
    with open("accuracy.txt", "w") as f:
        f.write(f"Accuracy: {acc.item():.4f}\n")
except Exception as e:
    print(f"Erreur lors de la sauvegarde des résultats : {e}")