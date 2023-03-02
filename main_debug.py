from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os
import argparse
import wandb
import math
from tqdm import tqdm
from models import *
from attack import *
from manager import *

TEACHER=1

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--teacher_lr', default=1e-5, type=float, help='teacher learning rate')
parser.add_argument('--lr_schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
parser.add_argument('--lr_factor', default=0.1, type=float, help='factor by which to decrease lr')
parser.add_argument('--loss', default='TRADES', type=str, help='loss function')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs for training')
parser.add_argument('--output', default = '', type=str, help='output subdirectory')
parser.add_argument('--model', default = 'MobileNetV2', type = str, help = 'student model name')
parser.add_argument('--model_path', default='', type=str, help='path of student net')
parser.add_argument('--teacher_model', default = 'ResNet18', type = str, help = 'teacher network model')
parser.add_argument('--teacher_path', default = '', type=str, help='path of teacher net being distilled')
parser.add_argument('--temp', default=30.0, type=float, help='temperature for distillation')
parser.add_argument('--val_period', default=1, type=int, help='print every __ epoch')
parser.add_argument('--save_period', default=1, type=int, help='save every __ epoch')
parser.add_argument('--alpha', default=1.0, type=float, help='weight for sum of losses')
parser.add_argument('--lambd', default=1.0/6.0, type=float, help='weight for KL item in TRADES')
parser.add_argument('--dataset', default = 'tiny_imagenet', type=str, help='name of dataset')
parser.add_argument('--debug', action='store_true', help='debug mode')
parser.add_argument('--resume_epoch', default=0, type=int)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max gradient norm')
parser.add_argument('--project_name', type=str, default='ARD-debug')
args = parser.parse_args()

def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(args.seed)

if args.output == '':
    args.output = '{}-{}'.format(args.model,args.dataset)
if args.debug:
    DEBUG = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == 'CIFAR10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    num_classes = 10
elif args.dataset == 'CIFAR100':
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    num_classes = 100


print('==> Building model..'+args.model)
if args.model == 'MobileNetV2':
	basic_net = MobileNetV2(num_classes=num_classes)
elif args.model == 'WideResNet':
	basic_net = WideResNet(num_classes=num_classes)
elif args.model == 'ResNet18':
	basic_net = ResNet18(num_classes=num_classes)
basic_net = basic_net.to(device)
if TEACHER:
	if args.teacher_model == 'MobileNetV2':
		teacher_net = MobileNetV2(num_classes=num_classes)
	elif args.teacher_model == 'WideResNet':
		teacher_net = WideResNet(num_classes=num_classes)
	elif args.teacher_model == 'ResNet18':
		teacher_net = ResNet18(num_classes=num_classes)
	teacher_net = teacher_net.to(device)

if args.model_path != '':
    print('==> Loading student model..')
    basic_net.load_state_dict(torch.load(args.model_path))

if args.teacher_path != '' and TEACHER:
    print('==> Loading teacher..')
    teacher_net.load_state_dict(torch.load(args.teacher_path))
    teacher_net.eval()

config = {
    'epsilon': 8.0 / 255,
    'num_steps': 5,
    'step_size': 2.0 / 255,
}
net = AttackPGD(basic_net, config)
adv_teacher_net = AttackPGD(teacher_net, config)
if device == 'cuda':
    cudnn.benchmark = True

KL_loss = nn.KLDivLoss()
XENT_loss = nn.CrossEntropyLoss()
lr=args.lr
num_steps=0
project_name=args.project_name

def train(epoch, optimizer, teacher_optimizer=None):
    global num_steps
    net.train()
    train_loss = 0
    adv_correct = 0
    natural_correct = 0
    total = 0
    iterator = tqdm(trainloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        num_steps += 1
        inputs, targets = inputs.to(device), targets.to(device)     
        optimizer.zero_grad()
        outputs, pert_inputs = net(inputs, targets)
        basic_outputs = basic_net(inputs)
        # ARD & KD need teacher basic output
        if args.loss == 'ARD':
            teacher_basic_outputs = teacher_net(inputs)
            loss = args.alpha*args.temp*args.temp*KL_loss(F.log_softmax(outputs/args.temp, dim=1),F.softmax(teacher_basic_outputs/args.temp, dim=1))+(1.0-args.alpha)*XENT_loss(basic_outputs, targets)
        elif args.loss == 'KD':
            teacher_basic_outputs = teacher_net(inputs)
            loss = args.alpha*args.temp*args.temp*KL_loss(F.log_softmax(basic_outputs/args.temp, dim=1),F.softmax(teacher_basic_outputs/args.temp, dim=1))+(1.0-args.alpha)*XENT_loss(basic_outputs, targets)
        elif args.loss == 'ARD-PRO':
            teacher_basic_outputs = teacher_net(inputs)
            teacher_outputs = teacher_net(pert_inputs)
            loss = args.alpha*args.temp*args.temp*KL_loss(F.log_softmax(outputs/args.temp, dim=1),F.softmax(teacher_basic_outputs/args.temp, dim=1))+(1.0-args.alpha)*XENT_loss(basic_outputs, targets)
            teacher_loss = XENT_loss(teacher_outputs, targets)
        # SAT & TRADES don't need teacher
        elif args.loss == 'SAT':
            loss = XENT_loss(outputs, targets)
        elif args.loss == 'TRADES':
            xent_loss = XENT_loss(basic_outputs, targets)
            kl_loss = args.temp*args.temp*KL_loss(F.log_softmax(outputs/args.temp, dim=1),F.softmax(basic_outputs/args.temp, dim=1))/args.lambd
            loss = xent_loss+kl_loss
        elif args.loss == 'NAT':
            loss = XENT_loss(basic_outputs, targets)
        
        if math.isnan(loss.item()):
            print('Error!')
        loss.backward()

   
        # Gradient monitor
        grad_list = []
        for param_group in optimizer.param_groups:
            params = param_group['params']
            mile_stone_1, mile_stone_2 = len(params) // 3, len(params) // 3 * 2
            for param in params:
                grad_list.append(torch.norm(param.grad.cpu(), p=2))
            break
        grads = np.array(grad_list)
        MG = float(grads.mean())
        MG_head = float(grads[0:mile_stone_1].mean())
        MG_inter = float(grads[mile_stone_1:mile_stone_2].mean())
        MG_tail = float(grads[mile_stone_2:].mean())
        if not DEBUG:
            wandb.log({
                'Student MG': MG,
                'Student Head MG': MG_head,
                'Student Intermediate MG': MG_inter,
                'Student Tail MG': MG_tail
            
            }, step=num_steps)
        
        torch.nn.utils.clip_grad_norm_(basic_net.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(teacher_net.parameters(), args.max_grad_norm)

        optimizer.step()
        if args.loss == 'ARD-PRO':
            teacher_loss.backward()
            teacher_optimizer.step()
        train_loss = loss.item()
        _, adv_predicted = outputs.max(1)
        _, natural_predicted = basic_outputs.max(1)
        natural_correct += natural_predicted.eq(targets).sum().item()
        total += targets.size(0)
        adv_correct += adv_predicted.eq(targets).sum().item()

        robust_acc = 100.*adv_correct/total
        natural_acc = 100.*natural_correct/total

        iterator.set_description('epoch {} | loss {:.5f} | racc {:.2f} | nacc {:.2f}'.format(
            epoch, loss.item(), robust_acc, natural_acc))
        if not DEBUG:
            wandb.log({'training_loss': loss.item()}, step=num_steps)
            if args.loss == 'ARD-PRO':
                wandb.log({'teacher_training_loss': teacher_loss.item()}, step=num_steps)
        if DEBUG and batch_idx > 9:
            break

    if (epoch+1)%args.save_period == 0:
        state = basic_net.state_dict()
        if not os.path.isdir(epoch_dir(args, epoch, project_name)):
            os.makedirs(epoch_dir(args, epoch, project_name))
        torch.save(state, epoch_dir(args, epoch, project_name)+'model_ckpt.t7')
        if args.loss == 'ARD-PRO':
            teacher_state = teacher_net.state_dict()
            torch.save(teacher_state, epoch_dir(args, epoch, project_name)+'teacher_ckpt.t7')
    if not DEBUG:
        wandb.log({'Natural train acc': natural_acc, 'Robust train acc': robust_acc}, step=num_steps)

    print('Epoch {} Mean Training Loss:'.format(epoch), train_loss/len(iterator), 
          'racc {:.2f} | nacc {:.2f}'.format(robust_acc, natural_acc))
    iterator.close()
    return train_loss

def test(epoch, optimizer):
    global num_steps
    net.eval()
    adv_correct = 0
    natural_correct = 0
    teacher_correct = 0
    adv_teacher_correct = 0
    adv_teacher_correct_self = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            adv_outputs, pert_inputs = net(inputs, targets)
            natural_outputs = basic_net(inputs)
            _, adv_predicted = adv_outputs.max(1)
            _, natural_predicted = natural_outputs.max(1)

            if args.loss == 'ARD-PRO':
                basic_teacher_output = teacher_net(inputs)
                adv_teacher_output = teacher_net(pert_inputs)
                adv_teacher_output_self, pert_teacher_inputs = adv_teacher_net(inputs, targets)
                _, teacher_predicted = basic_teacher_output.max(1)
                _, adv_teacher_predicted = adv_teacher_output.max(1)
                _, adv_teacher_predicted_self = adv_teacher_output_self.max(1)
                teacher_correct += teacher_predicted.eq(targets).sum().item()
                adv_teacher_correct += adv_teacher_predicted.eq(targets).sum().item()
                adv_teacher_correct_self += adv_teacher_predicted_self.eq(targets).sum().item()

            natural_correct += natural_predicted.eq(targets).sum().item()
            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()
            iterator.set_description(str(adv_predicted.eq(targets).sum().item()/targets.size(0)))

            if batch_idx > 9 and DEBUG:
                break

    robust_acc = 100.*adv_correct/total
    natural_acc = 100.*natural_correct/total
    print('Natural acc:', natural_acc)
    print('Robust acc:', robust_acc)
    natural_teacher_acc = 100.*teacher_correct/total
    adv_teacher_acc = 100.*adv_teacher_correct/total
    adv_teacher_acc_self = 100.*adv_teacher_correct_self/total
    print('Teacher acc:', natural_teacher_acc)
    print('Teacher robust acc:', adv_teacher_acc)
    print('Teacher robust acc self', adv_teacher_acc_self)
    if not DEBUG:
        wandb.log({'Natural test acc': natural_acc, 'Robust test acc': robust_acc}, step=num_steps)
        if args.loss == 'ARD-PRO':
            wandb.log({'Natural teacher acc': natural_teacher_acc, 
                       'Robust teacher acc': adv_teacher_acc,
                       'Robust teacher acc (self)': adv_teacher_acc_self
                    }, step=num_steps)
    iterator.close()
    return natural_acc, robust_acc

# 一个现象：DEBUG 模式下最初几个 epoch robust acc 下降，n acc 上升。

def main():
    if not DEBUG:
        wandb.init(project=project_name, name=args.output)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    if args.loss == 'ARD-PRO':
        teacher_optimizer = optim.SGD(teacher_net.parameters(), lr=args.teacher_lr, momentum=0.9, weight_decay=2e-4)
    best_val = 0
    manager = Manager(net, device, adversarial=True)
    for epoch in range(args.resume_epoch, args.epochs):
        adjust_learning_rate(args.lr, optimizer, epoch)
        adjust_teacher_learning_rate(args.teacher_lr, teacher_optimizer, epoch)
        if args.loss == 'ARD-PRO':
            train_loss = train(epoch, optimizer, teacher_optimizer)
        else:
            train_loss = train(epoch, optimizer)
        if (epoch+1)%args.val_period == 0:
            # current v.s. history best
            natural_val, robust_val = test(epoch, optimizer)
            if robust_val > best_val:
                best_val = robust_val
                # save best ckpt
                state = basic_net.state_dict()
                torch.save(state, current_dir(args, project_name)+'optimal_ckpt.t7'.format(epoch))

if __name__ == '__main__':
    main()
