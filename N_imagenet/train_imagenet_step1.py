import os
import torch
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.cuda import amp
from scipy.io import savemat
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from models.ReActNet_A_Qa import *
from models.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'

def SigmoidFunc(x, alpha=3.):   # Sigmoid function with parameter
    return torch.sigmoid(alpha * x)

def SigmoidGFunc(x, alpha=3.):   # Derivative of Sigmoid functions with parameter
    return alpha * SigmoidFunc(x, alpha) * (1 - SigmoidFunc(x, alpha))

def BinaryForFunc(x):   # Weight quantization forward function
    return torch.sign(x)

def BinaryBackFunc(x, alpha=3.):   # Quantization backward gradient function
    return 2 * SigmoidGFunc(x, alpha)

class SignFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return BinaryForFunc(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * BinaryBackFunc(input, alpha=3.)
    
class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()
        
    def forward(self, x):
        ba = SignFunc.apply(x)
        return ba

class Q_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
class Q_PReLU(nn.Module):
    def __init__(self, out_chn):
        super(Q_PReLU, self).__init__()
        self.L_alpha = nn.Parameter(-3 * torch.ones(out_chn), requires_grad=True)

    def forward(self, x):
        IL_alpha = Q_Func.apply(self.L_alpha)
        QIL_alpha = torch.pow(input = torch.tensor(2).to(x.device), exponent = IL_alpha)
        return F.prelu(x, QIL_alpha)

def get_weight(module):
    std, mean = torch.std_mean(module.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
    weight = (module.weight - mean) / (std + module.eps)
    return weight

def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class ScaledStdConv2d(nn.Conv2d):

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
            bias=False, gamma=1.0, eps=1e-5, use_layernorm=False):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps ** 2 if use_layernorm else eps
        self.use_layernorm = use_layernorm  # experimental, slightly faster/less GPU memory to hijack LN kernel

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        return self.gain * weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class HardBinaryScaledStdConv2d(nn.Module):

    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, gamma=1.0, eps=1e-5, use_layernorm=False):
        super(HardBinaryScaledStdConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand(self.shape) * 0.001, requires_grad=True)

        self.gain = nn.Parameter(torch.ones(out_chn, 1, 1, 1))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps ** 2 if use_layernorm else eps
        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)

        scaling_factor = torch.mean(torch.mean(torch.mean(abs(weight),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(weight)
        cliped_weights = torch.clamp(weight, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        return self.gain * binary_weights

    def forward(self, x):

        return F.conv2d(x, self.get_weight(), stride=self.stride, padding=self.padding)



#### Configuration"

Begin_epoch = 26
Max_epoch = 64
Learning_rate = 1e-3
Weight_decay = 5e-6
Momentum = 0.9
Top_k = 5
AMP = False
#REPRESENTATION_CHOICE = "_2_Channel_count"
REPRESENTATION_CHOICE = "_4_Channel_" 

Dataset_path = '../data/N-imagenet_preprocessed'+REPRESENTATION_CHOICE+'/'
train_path = os.path.join(Dataset_path, 'train')
test_path = os.path.join(Dataset_path, 'test')
Batch_size = 64
Workers = 8
Targetnum = 100

Test_every_iteration = None
Name_suffix = '_step1'
Savemodel_path = './savemodels/'
Record_path = './recorddata/'
if not os.path.exists(Savemodel_path):
    os.mkdir(Savemodel_path)
if not os.path.exists(Record_path):
    os.mkdir(Record_path)


_seed_ = 2023
torch.manual_seed(_seed_)
np.random.seed(_seed_)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Création des répertoires nécessaires
os.makedirs(Savemodel_path, exist_ok=True)
os.makedirs(Record_path, exist_ok=True)
os.makedirs(Dataset_path, exist_ok=True)

if REPRESENTATION_CHOICE == "_2_Channel_count":
    def from_events_to_img(events):
        """
        Convert the event stream to a 2D image using vectorized NumPy operations.
        Works with structured arrays with named fields (x, y, t, p).
        """
        # Create an empty image
        img = np.zeros((2, 448, 448), dtype=np.float32)
        
        # Filter valid events based on coordinates
        valid_mask = ((events['x'] >= 96) & 
                    (events['x'] < 544) & 
                    (events['y'] >= 16) & 
                    (events['y'] < 464))
        
        valid_events = events[valid_mask]
        
        # Extract components
        x = valid_events['x'] - 96
        y = valid_events['y'] - 16
        p = valid_events['p']
        
        # Separate positive and negative polarities
        pos_mask = p > 0
        neg_mask = ~pos_mask
        
        # Update image at respective coordinates
        np.add.at(img[0], (y[pos_mask], x[pos_mask]), 1.0)
        np.add.at(img[1], (y[neg_mask], x[neg_mask]), 1.0)
        
        # Normalize each channel efficiently
        img /= np.maximum(np.max(np.abs(img), axis=(1, 2)), 1e-3)[:, np.newaxis, np.newaxis]
        
        return img
elif REPRESENTATION_CHOICE == "_4_Channel_":
    def from_events_to_img(events):
        """
        Convert the event stream to a 4-channel 2D image using vectorized NumPy operations.
        Works with structured arrays with named fields (x, y, t, p).
        """
        import numpy as np
        
        # Image dimensions
        H, W = 448, 448
        
        # Create an empty image (4 channels: pos_count, pos_time, neg_count, neg_time)
        img = np.zeros((4, H, W), dtype=np.float32)
        
        # Filter valid events based on coordinates
        valid_mask = ((events['x'] >= 96) &
                    (events['x'] < 544) &
                    (events['y'] >= 16) &
                    (events['y'] < 464))
        valid_events = events[valid_mask]
        
        # Extract components and adjust coordinates
        x = valid_events['x'] - 96
        y = valid_events['y'] - 16
        p = valid_events['p']
        t = valid_events['t']
        
        # Get time normalization values
        start_time = t[0] if len(t) > 0 else 0
        time_length = t[-1] - start_time if len(t) > 0 and t[-1] > start_time else 1.0
        
        # Separate positive and negative polarities
        pos_mask = p > 0
        neg_mask = ~pos_mask
        
        # Handle positive events
        pos_y, pos_x = y[pos_mask], x[pos_mask]
        pos_t = (t[pos_mask] - start_time) / time_length  # Normalize timestamps
        
        # Handle negative events
        neg_y, neg_x = y[neg_mask], x[neg_mask]
        neg_t = (t[neg_mask] - start_time) / time_length  # Normalize timestamps
        
        # Count events (channel 0 and 2)
        np.add.at(img[0], (pos_y, pos_x), 1.0)  # Positive count
        np.add.at(img[2], (neg_y, neg_x), 1.0)  # Negative count
        
        # Normalize counts
        pos_max = np.max(img[0])
        neg_max = np.max(img[2])
        img[0] /= max(pos_max, 1e-3)
        img[2] /= max(neg_max, 1e-3)
        
        # Create temporary arrays for max timestamp calculation
        pos_time_img = np.zeros((H, W), dtype=np.float32)
        neg_time_img = np.zeros((H, W), dtype=np.float32)
        
        # This mimics scatter_max by using a loop (vectorizing this is challenging in NumPy)
        for i in range(len(pos_y)):
            py, px = pos_y[i], pos_x[i]
            pos_time_img[py, px] = max(pos_time_img[py, px], pos_t[i])
        
        for i in range(len(neg_y)):
            ny, nx = neg_y[i], neg_x[i]
            neg_time_img[ny, nx] = max(neg_time_img[ny, nx], neg_t[i])
        
        # Assign timestamps to channels 1 and 3
        img[1] = pos_time_img
        img[3] = neg_time_img
        
        return img

def process_dataset(input_base_dir, output_base_dir):
    """
    Process both train and test datasets
    """
    # Traiter le jeu d'entraînement
    input_train_dir = os.path.join(input_base_dir, "train")
    output_train_dir = os.path.join(output_base_dir, "train")
    process_all_files(input_train_dir, output_train_dir)
    
    # Traiter le jeu de test
    input_test_dir = os.path.join(input_base_dir, "test")
    output_test_dir = os.path.join(output_base_dir, "test")
    process_all_files(input_test_dir, output_test_dir)

def process_all_files(input_dir, output_dir):
    """
    Process all .npz files in the input directory structure and save results
    in the same structure in the output directory.
    """
    # Créer le répertoire de sortie s'il n'existe pas
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Parcourir tous les sous-dossiers (classes)
    class_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir()]
    print(f"Traitement de {len(class_dirs)} classes...")
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        output_class_dir = Path(output_dir) / class_name
        output_class_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Traitement de la classe: {class_name}")
        
        # Parcourir tous les fichiers .npz dans cette classe
        npz_files = list(class_dir.glob("*.npz"))
        
        for npz_file in tqdm(npz_files, desc=f"Classe {class_name}"):
            try:
                # Charger le fichier .npz
                data = np.load(str(npz_file))
                
                # Détecter le nom de la clé d'événements
                event_key = 'event_data'
                event_stream = data[event_key]
                
                img = from_events_to_img(event_stream)
                
                # Créer le fichier de sortie avec le même nom
                output_file = output_class_dir / npz_file.name
                
                # Sauvegarder l'image transformée
                np.savez_compressed(
                    str(output_file),
                    image=img,
                    label=class_name,
                    allow_pickle=True,
                )
            except Exception as e:
                print(f"Erreur lors du traitement de {npz_file}: {e}")
    
    print(f"Traitement terminé pour {input_dir} -> {output_dir}")

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

if os.path.exists(train_path) and os.path.exists(test_path):
    print("Les données prétraitées existent déjà.")
else:
    print("Prétraitement des données...")
    input_dir = "N-imagenet/"
    process_dataset(input_dir, Dataset_path)
        
    # Exemple d'utilisation du dataset (décommentez pour tester)
train_dataset = NImageNetDataset(train_path)
test_dataset = NImageNetDataset(test_path)

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=Batch_size, shuffle=True, num_workers=Workers, pin_memory=True#, drop_last=True
)
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=Batch_size, shuffle=False, num_workers=Workers, pin_memory=True#, drop_last=False
)

### MODEL DEFINITION


stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2

def Bconv3x3(in_planes, out_planes, stride=1):
    """3x3 binary convolution with padding"""
    return HardBinaryScaledStdConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

def Bconv1x1(in_planes, out_planes, stride=1):
    """1x1 binary convolution without padding"""
    return HardBinaryScaledStdConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

def conv3x3(in_planes, out_planes, stride=1):
    return ScaledStdConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return ScaledStdConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x, beta=1):
        out = x + self.bias.expand_as(x) / beta
        return out

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()
        self.conv1 = ScaledStdConv2d(inp, oup, 3, stride, 1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, alpha, beta1, beta2, stride=1):
        super(BasicBlock, self).__init__()

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2 

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3 = conv3x3(inplanes, inplanes, stride=stride)
        self.move12 = LearnableBias(inplanes)
        
#         self.prelu1 = nn.PReLU(inplanes)
#         self.prelu1 = nn.LeakyReLU(0.125)
        self.prelu1 = Q_PReLU(inplanes)
        
        self.move13 = LearnableBias(inplanes)
        self.move21 = LearnableBias(inplanes)
        
        if inplanes == planes:
            self.binary_pw = conv1x1(inplanes, planes)
        else:
            self.binary_pw_down1 = conv1x1(inplanes, inplanes)
            self.binary_pw_down2 = conv1x1(inplanes, inplanes)
            
        self.move22 = LearnableBias(planes)
        
#         self.prelu2 = nn.PReLU(planes)
#         self.prelu2 = nn.LeakyReLU(0.125)
        self.prelu2 = Q_PReLU(planes)
    
        self.move23 = LearnableBias(planes)
        
        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2,2)

    def forward(self, x):
        
        x_in = x

        out1 = self.move11(x_in, self.beta1)
        out1 = self.binary_activation(out1 * self.beta1)
        out1 = self.binary_3x3(out1)

        if self.stride == 2:
            x = self.pooling(x_in)

        out1 = x + out1*self.alpha

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)
        
        out1_in = out1
        out2 = self.move21(out1_in, self.beta2)
        out2 = self.binary_activation(out2 * self.beta2)
    
        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = out2*self.alpha + out1
        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = out2_1*self.alpha + out1
            out2_2 = out2_2*self.alpha + out1
            out2 = torch.cat([out2_1, out2_2], dim=1)
            
        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2

class Reactnet(nn.Module):

    def __init__(self, alpha=0.25, num_classes=1000, imagenet=True):
        super(Reactnet, self).__init__()
        
        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:
                expected_var = 1.0
                beta1 = 1. / expected_var ** 0.5
                expected_var += alpha ** 2
                beta2 = 1. / expected_var ** 0.5
                if imagenet:
                    self.feature.append(firstconv3x3(4, stage_out_channel[i], 2))
                else:
                    self.feature.append(firstconv3x3(4, stage_out_channel[i], 2))
            elif stage_out_channel[i-1] != stage_out_channel[i]:# and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], alpha, beta1, beta2, 2))
                # Reset expected var at a transition block
                expected_var = 1.0
                beta1 = 1. / expected_var ** 0.5
                expected_var += alpha ** 2
                beta2 = 1. / expected_var ** 0.5

            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], alpha, beta1, beta2, 1))
                expected_var += alpha ** 2
                beta1 = 1. / expected_var ** 0.5
                expected_var += alpha ** 2
                beta2 = 1. / expected_var ** 0.5

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        for _, block in enumerate(self.feature):
            x = block(x)

        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
net = Reactnet(num_classes=Targetnum, imagenet=True)

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

# Parameters 
all_parameters = net.parameters()
weight_parameters = []
for pname, p in net.named_parameters():
    if (p.ndimension() == 4 or 'conv' in pname) and 'L_alpha' not in pname:
        weight_parameters.append(p)
weight_parameters_id = list(map(id, weight_parameters))
other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

#criterion_train = DistributionLoss()
criterion_test = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params' : other_parameters, 'weight_decay' : 0., 'initial_lr': Learning_rate},
    {'params' : weight_parameters, 'weight_decay' : Weight_decay, 'initial_lr': Learning_rate}],
    lr = Learning_rate)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/Max_epoch), last_epoch=Begin_epoch-1)

### train

def test_model(net, max_test_acc, data_loader=test_data_loader, criterion=criterion_test, epoch=None, iteration=None, record=True):
    net.eval()
    test_samples = 0
    test_loss = 0
    test_acc_top1 = 0
    test_acc_topk = 0
    
    with torch.no_grad():
        for img, label in tqdm(data_loader):
            img = img.cuda()
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

def train_model(net, max_test_acc, epoch, data_loader=train_data_loader, optimizer=optimizer, criterion=criterion_test, scaler=scaler, record=True):
    train_samples = 0
    train_loss = 0
    train_acc_top1 = 0
    train_acc_topk = 0
    
    for i, (img, label) in enumerate(tqdm(data_loader)):
        net.train()
        img = img.cuda()
        label = label.cuda()
        label_onehot = F.one_hot(label, Targetnum).float()
        
        out_fr = net(img)
        out_teacher = label#model_teacher(img)
        loss = criterion(out_fr, out_teacher)
            
        train_samples += label.numel()
        train_loss += loss.item() * label.numel()

        train_acc_top1 += (out_fr.argmax(1) == label).float().sum().item()
        _, pred = out_fr.topk(Top_k, 1, True, True)
        train_acc_topk += torch.eq(pred, label.view(-1,1)).float().sum().item()
        
        optimizer.zero_grad()
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
        img = img.cuda()
        label = label.cuda()
        out_fr = net(img)
        guess = out_fr.argmax(1)
        for j in range(len(label)):
            Confusion_Matrix[label[j],guess[j]] += 1
acc = Confusion_Matrix.diag()
acc = acc.sum()/Confusion_Matrix.sum()


try:
    torch.save(Confusion_Matrix.cpu(), Record_path+"confusion_matrix_step1.pt")
    pd.DataFrame(Confusion_Matrix.cpu().numpy()).to_csv(Record_path+"confusion_matrix_step1.csv", index=False)
    with open("accuracy_step1.txt", "w") as f:
        f.write(f"Accuracy: {acc.item():.4f}\n")
except Exception as e:
    print(f"Erreur lors de la sauvegarde des résultats : {e}")






####STEP 2
#changement avec QAW

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2

def Bconv3x3(in_planes, out_planes, stride=1):
    """3x3 binary convolution with padding"""
    return HardBinaryScaledStdConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

def Bconv1x1(in_planes, out_planes, stride=1):
    """1x1 binary convolution without padding"""
    return HardBinaryScaledStdConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)

def conv3x3(in_planes, out_planes, stride=1):
    return ScaledStdConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return ScaledStdConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x, beta=1):
        out = x + self.bias.expand_as(x) / beta
        return out

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()
        self.conv1 = ScaledStdConv2d(inp, oup, 3, stride, 1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, alpha, beta1, beta2, stride=1):
        super(BasicBlock, self).__init__()

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2 

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3 = Bconv3x3(inplanes, inplanes, stride=stride)
        self.move12 = LearnableBias(inplanes)
        
#         self.prelu1 = nn.PReLU(inplanes)
#         self.prelu1 = nn.LeakyReLU(0.125)
        self.prelu1 = Q_PReLU(inplanes)
        
        self.move13 = LearnableBias(inplanes)
        self.move21 = LearnableBias(inplanes)
        
        if inplanes == planes:
            self.binary_pw = Bconv1x1(inplanes, planes)
        else:
            self.binary_pw_down1 = Bconv1x1(inplanes, inplanes)
            self.binary_pw_down2 = Bconv1x1(inplanes, inplanes)
            
        self.move22 = LearnableBias(planes)
        
#         self.prelu2 = nn.PReLU(planes)
#         self.prelu2 = nn.LeakyReLU(0.125)
        self.prelu2 = Q_PReLU(planes)
    
        self.move23 = LearnableBias(planes)
        
        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2,2)

    def forward(self, x):
        
        x_in = x

        out1 = self.move11(x_in, self.beta1)
        out1 = self.binary_activation(out1 * self.beta1)
        out1 = self.binary_3x3(out1)

        if self.stride == 2:
            x = self.pooling(x_in)

        out1 = x + out1*self.alpha

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)
        
        out1_in = out1
        out2 = self.move21(out1_in, self.beta2)
        out2 = self.binary_activation(out2 * self.beta2)
    
        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = out2*self.alpha + out1
        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = out2_1*self.alpha + out1
            out2_2 = out2_2*self.alpha + out1
            out2 = torch.cat([out2_1, out2_2], dim=1)
            
        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2

class Reactnet(nn.Module):

    def __init__(self, alpha=0.25, num_classes=1000, imagenet=True):
        super(Reactnet, self).__init__()
        
        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:
                expected_var = 1.0
                beta1 = 1. / expected_var ** 0.5
                expected_var += alpha ** 2
                beta2 = 1. / expected_var ** 0.5
                if imagenet:
                    self.feature.append(firstconv3x3(4, stage_out_channel[i], 2))
                else:
                    self.feature.append(firstconv3x3(4, stage_out_channel[i], 2))
            elif stage_out_channel[i-1] != stage_out_channel[i]:# and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], alpha, beta1, beta2, 2))
                # Reset expected var at a transition block
                expected_var = 1.0
                beta1 = 1. / expected_var ** 0.5
                expected_var += alpha ** 2
                beta2 = 1. / expected_var ** 0.5

            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], alpha, beta1, beta2, 1))
                expected_var += alpha ** 2
                beta1 = 1. / expected_var ** 0.5
                expected_var += alpha ** 2
                beta2 = 1. / expected_var ** 0.5

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        for _, block in enumerate(self.feature):
            x = block(x)

        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



Name_suffix = '_step2'
Begin_epoch = 0
Max_epoch = 64
Learning_rate = 1e-3
Weight_decay = 0
Momentum = 0.9
Top_k = 5


### MODEL DEFINITION
net = Reactnet(num_classes=Targetnum, imagenet=True)


net = nn.DataParallel(net).cuda()
max_test_acc = 0.
if Begin_epoch!=0:
    net.load_state_dict(torch.load(Savemodel_path + f'epoch{Begin_epoch-1}{Name_suffix}.h5'))
    max_test_acc = np.load(Savemodel_path + f'max_acc{Name_suffix}.npy')
    max_test_acc = max_test_acc.item()
else:
    net.load_state_dict(torch.load(Savemodel_path + f'max_acc_step1.h5'))

scaler = amp.GradScaler() if AMP else None
Test_top1 = []
Test_topk = []
Test_lossall = []
Epoch_list = []
Iteration_list = []

criterion_test = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params' : net.parameters(), 'weight_decay' : Weight_decay, 'initial_lr': Learning_rate}],
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
            img = img.cuda()
            label = label.cuda()
            
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


def train_model(net, max_test_acc, epoch, data_loader=train_data_loader, optimizer=optimizer, criterion=criterion_test, scaler=scaler, record=True):
    train_samples = 0
    train_loss = 0
    train_acc_top1 = 0
    train_acc_topk = 0
    
    for i, (img, label) in enumerate(tqdm(data_loader)):
        net.train()
        img = img.cuda()
        label = label.cuda()
        
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
        img = img.cuda()
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