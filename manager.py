import os
import torch
import random 
import argparse
import numpy as np
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
from pprint import pprint

from tqdm import tqdm

DEBUG=0

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--teacher_lr', default=1e-5, type=float, help='teacher learning rate')
    parser.add_argument('--lr_schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
    parser.add_argument('--lr_factor', default=0.1, type=float, help='factor by which to decrease lr')
    parser.add_argument('--loss', default='TRADES', type=str, help='loss function')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs for training')
    parser.add_argument('--output', default = '', type=str, help='output subdirectory')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--model', default = 'MobileNetV2', type = str, help = 'student model name')
    parser.add_argument('--model_path', default='', type=str, help='student model checkpoint')
    parser.add_argument('--teacher_model', default = 'ResNet18', type = str, help = 'teacher network model')
    parser.add_argument('--teacher_path', default = '', type=str, help='path of teacher net being distilled')

    parser.add_argument('--temp', default=1., type=float, help='temperature for distillation')
    parser.add_argument('--val_period', default=1, type=int, help='print every __ epoch')
    parser.add_argument('--save_period', default=1, type=int, help='save every __ epoch')
    parser.add_argument('--alpha', default=1.0, type=float, help='weight for sum of losses')
    parser.add_argument('--lambd', default=1.0, type=float, help='weight for KL item in TRADES')
    parser.add_argument('--dataset', default = 'tiny_imagenet', type=str, help='name of dataset')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max gradient norm')
    parser.add_argument('--project_name', type=str, default='ARD-debug')
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--eval_only', action='store_true')

    parser.add_argument('--memorization', type=int, default=0, help='teacher memorization loss')
    parser.add_argument('--aux_loss', type=str, default='SAT', help='teacher auxiliary loss')
    parser.add_argument('--aux_alpha', type=str, default='MOST', help='teacher sample wise attention strategy')
    parser.add_argument('--aux_lamda', type=float, default=1.0, help='teacher aux loss adjust factor')
    args = parser.parse_args()

    pprint(vars(args))    
    return args

def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def adjust_learning_rate(lr, optimizer, epoch):
    if epoch >= 90:
        lr *= 0.001
    elif epoch >= 70:
        lr *= 0.01
    elif epoch >= 50:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_teacher_learning_rate(origin_lr, optimizer, epoch):
    lr = 0.
    if epoch >= 70:
        lr = origin_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def plot_histgram(path, results):
    plt.clf()
    sns.histplot(results)
    plt.savefig(path, dpi=400, format='svg')

def current_dir(args, project='distill_project'):
    return '/root/checkpoint/{}/'.format(project)+args.dataset+'/'+args.output+'/'

def epoch_dir(args, epoch, project='distill_project'):
    return current_dir(args, project)+'epoch'+str(epoch)+'/'

def should_teacher_tune(loss_name):
    registered_loss_names = [
        'ARD-PRO',
        'COMB',
        'KL-Coarse',
    ]
    return (loss_name in registered_loss_names)

class Manager():
    def __init__(self, net, device, adversarial = False):
        self.net = net 
        self.device = device
        self.adversarial = adversarial
        self.results = None

    def collect(self, dataloader):
        # 搜集正确类别/平均 logit，正确类别梯度/平均梯度
        gap_list = []
        intense_list = []
        cos_list = []
        
        with torch.enable_grad():
            iterator = tqdm(dataloader, ncols=0, leave=False, desc='collecting and analysing gradients')
            for batch_idx, (inputs, targets) in enumerate(iterator):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.adversarial:
                    outputs, inputs = self.net(inputs, targets)
                else:
                    outputs = self.net(inputs)

                iterator.set_description('collecting gradients')
                sample_num = targets.shape[0]
                
                for sample_idx in range(sample_num):
            
                    # logit
                    true_label = targets[sample_idx]
                    logit = outputs[sample_idx, true_label]
                    avg_logit = torch.mean(outputs[sample_idx])

                    # grad
                    # 注意 1. autograd.grad() 里面的参数用 inputs 而非 inputs[sample_idx] 否则会报错
                    # Attack 内部返回的 x 记得先 retains_grad_()
                    grad = torch.autograd.grad(logit, [inputs], retain_graph=True)[0][sample_idx]
                    avg_grad = torch.autograd.grad(avg_logit, [inputs], retain_graph=True)[0][sample_idx]
                    intensity = torch.norm(avg_grad,p=2)
                    cos = (grad*avg_grad).sum() / (torch.norm(grad,p=2)*intensity)

                    # collect
                    gap_list.append(abs(float(logit-avg_logit)))
                    intense_list.append(float(intensity))
                    cos_list.append(float(cos))

                if DEBUG:
                    if batch_idx >= 9:
                        break
        
        self.results = {
            'logit_diff': gap_list,
            'grad_intensity': intense_list,
            'cosine_relation': cos_list
        }

    def plot(self, output_dir):
        for grad_vis, grad_list in self.results.items():
            plot_histgram(path=os.path.join(output_dir, grad_vis + '.svg'), results=grad_list)
        self.results = None
