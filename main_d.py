from __future__ import print_function
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import os
import wandb
from tqdm import tqdm
from models import *
from attack import *
from manager import *
from data import *
from method import *

TEACHER=0
args = parse_args()
set_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainloader, testloader, num_classes = get_loader(args.dataset)

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

EPSILON = 8.0 / 255
config = {
    'epsilon': EPSILON,
    'num_steps': 5,
    'step_size': 2.0 / 255,
}
net = AttackPGD(basic_net, config)
if TEACHER:
    adv_teacher_net = AttackPGD(teacher_net, config)
if device == 'cuda':
    cudnn.benchmark = True

if args.model_path != '':
    print('==> Loading student..')
    basic_net.load_state_dict(torch.load(args.model_path))
if args.teacher_path != '':
    print('==> Loading teacher..')
    teacher_net.load_state_dict(torch.load(args.teacher_path))
    teacher_net.eval()

# output and epochs
if args.output == '':
    args.output = '{}-{}'.format(args.model,args.dataset)
if args.debug:
    DEBUG = 1
num_steps=args.resume_epoch*len(trainloader)
if DEBUG:
    args.epochs = args.resume_epoch+2


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

        # FOLLOWING: only for pretraining teacher
        # SAT & TRADES don't need teacher
        if args.loss == 'SAT':
            loss = XENTLoss.cal(outputs, targets)
        elif args.loss == 'TRADES':
            loss = TRADESLoss.cal(basic_outputs, outputs, targets, args.temp, args.lambd)
        elif args.loss == 'NAT':
            loss = XENTLoss.cal(basic_outputs, targets)
        loss.backward()


        # Gradient monitor
        if should_teacher_tune(args.loss):
            teacher_grad_list = []
            for param_group in teacher_optimizer.param_groups:
                params = param_group['params']
                for param in params:
                    teacher_grad_list.append(torch.norm(param.grad.cpu(), p=2))
            teacher_grads = np.array(teacher_grad_list)
            if not DEBUG:
                wandb.log({
                    'Teacher MG': float(teacher_grads.mean())
                }, step=num_steps)
            
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


        # Gradient Clip (Glip)
        torch.nn.utils.clip_grad_norm_(basic_net.parameters(), args.max_grad_norm)

        optimizer.step()
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
        if DEBUG:
            if batch_idx >= 1: break

    if not DEBUG:
        if (epoch+1)%args.save_period == 0:
            state = basic_net.state_dict()
            if not os.path.isdir(epoch_dir(args, epoch, args.project_name)):
                os.makedirs(epoch_dir(args, epoch, args.project_name))
            torch.save(state, epoch_dir(args, epoch, args.project_name)+'model_ckpt.t7')
            if should_teacher_tune(args.loss):
                teacher_state = teacher_net.state_dict()
                torch.save(teacher_state, epoch_dir(args, epoch, args.project_name)+'teacher_ckpt.t7')

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
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            adv_outputs, pert_inputs = net(inputs, targets)
            natural_outputs = basic_net(inputs)
            _, adv_predicted = adv_outputs.max(1)
            _, natural_predicted = natural_outputs.max(1)

            natural_correct += natural_predicted.eq(targets).sum().item()
            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()
            iterator.set_description(str(adv_predicted.eq(targets).sum().item()/targets.size(0)))
            if DEBUG:
                if batch_idx >= 9: break
            
    robust_acc = 100.*adv_correct/total
    natural_acc = 100.*natural_correct/total
    print('Natural acc:', natural_acc)
    print('Robust acc:', robust_acc)

    if not DEBUG and not args.eval_only:
        wandb.log({'Natural test acc': natural_acc, 
                   'Robust test acc': robust_acc,
                }, step=num_steps)
    iterator.close()
    return natural_acc, robust_acc

# 一个现象：DEBUG 模式下最初几个 epoch robust acc 下降，n acc 上升。 

def main():
    if not DEBUG:
        wandb.init(project=args.project_name, name=args.output)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)
    best_val = 0
    manager = Manager(net, device, adversarial=True)
    test(-1, None)
    for epoch in range(args.resume_epoch, args.epochs):
        adjust_learning_rate(args.lr, optimizer, epoch)
        train_loss = train(epoch, optimizer)
        if (epoch+1)%args.val_period == 0:
            # current v.s. history best
            natural_val, robust_val = test(epoch, optimizer)
            if robust_val > best_val and not DEBUG:
                best_val = robust_val
                # save best ckpt
                state = basic_net.state_dict()
                torch.save(state, current_dir(args, args.project_name)+'optimal_ckpt.t7'.format(epoch))

if __name__ == '__main__':
    if args.eval_only:
        args.loss = ''
        test(0, None)
    else:
        main()
