from __future__ import print_function
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import os
import copy
import wandb
from tqdm import tqdm
from models import *
from attack import *
from manager import *
from data import *
from method import *

TEACHER=1
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

        # FOLLOWING: for distilling student with teacher
        # ARD & KD need teacher basic output
        if args.loss == 'ARD':
            teacher_basic_outputs = teacher_net(inputs)
            loss = ARDLoss.cal(basic_outputs, outputs, teacher_basic_outputs, targets, args.alpha, args.temp)
        elif args.loss == 'KD':
            teacher_basic_outputs = teacher_net(inputs)
            loss = KDLoss.cal(basic_outputs, teacher_basic_outputs, targets, args.alpha, args.temp)
        elif should_teacher_tune(args.loss):
            teacher_basic_outputs = teacher_net(inputs)
            teacher_outputs = teacher_net(pert_inputs)
            if args.loss == 'ARD-PRO':
                loss, teacher_loss = ARDPROLoss.cal(basic_outputs, outputs, teacher_basic_outputs, teacher_outputs, targets, args.alpha, args.temp)
            elif args.loss == 'KL-Coarse':
                loss, teacher_loss = KLCoarseLoss.cal(basic_outputs, outputs, teacher_basic_outputs, teacher_outputs, targets, args.alpha, args.temp)
            elif args.loss == 'COMB':
                # inner utils
                
                if args.aux_alpha == 'MOST':
                    alpha_factor = AlphaFactorMost.cal(teacher_outputs, targets) 
                elif args.aux_alpha == 'LEAST':
                    alpha_factor = AlphaFactorLeast.cal(teacher_outputs, targets)
                elif args.aux_alpha == 'TargetSE':
                    alpha_factor = AlphaFactorTargetSE.cal(teacher_basic_outputs, teacher_outputs, targets)
                elif args.aux_alpha == 'SE':
                    alpha_factor = AlphaFactorSE.cal(teacher_basic_outputs, teacher_outputs)
                elif args.aux_alpha == 'FOSC':
                    grad = fosc_deps(pert_inputs, teacher_outputs, targets)
                    alpha_factor = AlphaFOSC.cal(EPSILON, grad, pert_inputs, inputs)
                elif args.aux_alpha == 'invFOSC':
                    grad = fosc_deps(pert_inputs, teacher_outputs, targets)
                    alpha_factor = AlphainvFOSC.cal(EPSILON, grad, pert_inputs, inputs)

                if args.aux_loss == 'SAT':
                    aux_loss = XENTLoss.cal(teacher_outputs, targets, no_reduction=True)
                elif args.aux_loss == 'KL':
                    aux_loss = KLoss.cal(teacher_basic_outputs, teacher_outputs, args.temp, no_reduction=True)
                elif args.aux_loss == 'SE':
                    aux_loss = SELoss.cal(teacher_basic_outputs, teacher_outputs, no_reduction=True)
                
                if not DEBUG:
                    wandb.log(
                        {'Mean alpha': float(torch.mean(alpha_factor)),
                         'Max alpha': float(torch.max(alpha_factor)),
                         'Min alpha': float(torch.min(alpha_factor))
                    }, step=num_steps)
                else:
                    pprint({'Mean alpha': float(torch.mean(alpha_factor)),
                         'Max alpha': float(torch.max(alpha_factor)),
                         'Min alpha': float(torch.min(alpha_factor))
                    })

                alpha_factor = alpha_factor.detach()
                alpha_factor = rank(alpha_factor)
                teacher_loss = (aux_loss * alpha_factor).mean()
                loss = ARDLoss.cal(basic_outputs, outputs, teacher_basic_outputs, targets, args.alpha, args.temp)

            teacher_loss *= args.aux_lamda
            if args.memorization:
                teacher_self_outputs, _ = adv_teacher_net(inputs, targets)
                memo_loss = TRADESLoss.cal(teacher_basic_outputs, teacher_self_outputs, targets, args.temp, args.lambd)
                teacher_loss += memo_loss

        # FOLLOWING: only for pretraining teacher
        # SAT & TRADES don't need teacher
        # elif args.loss == 'SAT':
        #     loss = XENT_loss(outputs, targets)
        # elif args.loss == 'TRADES':
        #     loss = TRADESLoss.cal(basic_outputs, outputs, targets, args.temp, args.lambd)
        # elif args.loss == 'NAT':
        #     loss = XENT_loss(basic_outputs, targets)
        if should_teacher_tune(args.loss):
            teacher_loss.backward(retain_graph=True)
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


        # Gradient Clip (Glip)
        torch.nn.utils.clip_grad_norm_(basic_net.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(teacher_net.parameters(), args.max_grad_norm)

        optimizer.step()
        if should_teacher_tune(args.loss):
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
            if should_teacher_tune(args.loss):
                wandb.log({'teacher_training_loss': teacher_loss.item()}, step=num_steps)
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

perts = []

def bench_test():
    student_correct = 0
    teacher_correct = 0
    total = 0
    for pert_inputs, targets in perts:
        teacher_outputs = teacher_net(pert_inputs)
        student_outputs = basic_net(pert_inputs)
        _, teacher_predicted = teacher_outputs.max(1)
        _, student_predicted = student_outputs.max(1)
        teacher_correct += teacher_predicted.eq(targets).sum().item()
        student_correct += student_predicted.eq(targets).sum().item()
        total += targets.size(0)
    return 100.*teacher_correct/total, 100.*student_correct/total


def test(epoch, optimizer):
    global num_steps
    net.eval()
    adv_correct = 0
    natural_correct = 0
    teacher_correct = 0
    adv_teacher_correct = 0
    adv_teacher_correct_self = 0
    adv_student_correct_cross = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            adv_outputs, pert_inputs = net(inputs, targets)
            natural_outputs = basic_net(inputs)
            _, adv_predicted = adv_outputs.max(1)
            _, natural_predicted = natural_outputs.max(1)

            # if should_teacher_tune(args.loss):
            basic_teacher_output = teacher_net(inputs)
            adv_teacher_output = teacher_net(pert_inputs)
            adv_teacher_output_self, teacher_pert_inputs = adv_teacher_net(inputs, targets)
            _, teacher_predicted = basic_teacher_output.max(1)
            _, adv_teacher_predicted = adv_teacher_output.max(1)
            _, adv_teacher_predicted_self = adv_teacher_output_self.max(1)
            teacher_correct += teacher_predicted.eq(targets).sum().item()
            adv_teacher_correct += adv_teacher_predicted.eq(targets).sum().item()
            adv_teacher_correct_self += adv_teacher_predicted_self.eq(targets).sum().item()

            # student robust acc on teacher pert inputs
            student_adv_outputs_cross = basic_net(teacher_pert_inputs)
            _, student_adv_preds_cross = student_adv_outputs_cross.max(1)
            adv_student_correct_cross += student_adv_preds_cross.eq(targets).sum().item()

            natural_correct += natural_predicted.eq(targets).sum().item()
            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()
            iterator.set_description(str(adv_predicted.eq(targets).sum().item()/targets.size(0)))
            if DEBUG:
                if batch_idx >= 9: break
            
            if epoch < 0:
                perts.append((
                    teacher_pert_inputs.clone().detach(), 
                    targets.clone().detach()
                ))

    robust_acc = 100.*adv_correct/total
    natural_acc = 100.*natural_correct/total
    robsut_acc_cross = 100.*adv_student_correct_cross/total
    print('Natural acc:', natural_acc)
    print('Robust acc:', robust_acc)
    print('Cross robust acc:', robsut_acc_cross)

    # if should_teacher_tune(args.loss):
    natural_teacher_acc = 100.*teacher_correct/total
    adv_teacher_acc = 100.*adv_teacher_correct/total
    adv_teacher_acc_self = 100.*adv_teacher_correct_self/total
    print('Teacher acc:', natural_teacher_acc)
    print('Teacher robust acc:', adv_teacher_acc)
    print('Teacher robust acc self', adv_teacher_acc_self)

    if epoch >= 0 and len(perts) > 0:
        bench_teacher_acc, bench_student_acc = bench_test()
        print('Benchmark teacher ', bench_teacher_acc)
        print('Benchmark student ', bench_student_acc)

    if not DEBUG and not args.eval_only:
        wandb.log({'Natural test acc': natural_acc, 
                   'Robust test acc': robust_acc,
                   'Cross robust acc': robsut_acc_cross,
                }, step=num_steps)
        # if should_teacher_tune(args.loss):
        wandb.log({'Natural teacher acc': natural_teacher_acc, 
                    'Robust teacher acc': adv_teacher_acc,
                    'Robust teacher acc (self)': adv_teacher_acc_self
                }, step=num_steps)
        if epoch >= 0 and len(perts) > 0:
            wandb.log({
                'Benchmark Student': bench_student_acc,
                'Benchmark Teacher': bench_teacher_acc
            })
    iterator.close()
    return natural_acc, robust_acc

# 一个现象：DEBUG 模式下最初几个 epoch robust acc 下降，n acc 上升。 

def main():
    if not DEBUG:
        wandb.init(project=args.project_name, name=args.output)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)
    if should_teacher_tune(args.loss):
        teacher_optimizer = optim.SGD(teacher_net.parameters(), lr=args.teacher_lr, momentum=0.9, weight_decay=2e-4)
    best_val = 0
    manager = Manager(net, device, adversarial=True)
    test(-1, None)
    for epoch in range(args.resume_epoch, args.epochs):
        adjust_learning_rate(args.lr, optimizer, epoch)
        if should_teacher_tune(args.loss):
            adjust_teacher_learning_rate(args.teacher_lr, teacher_optimizer, epoch)
        if should_teacher_tune(args.loss):
            train_loss = train(epoch, optimizer, teacher_optimizer)
        else:
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
    if False:
        from vis import *
        os.makedirs("./fig", exist_ok = True)
        visualize_metric_vs_prob(net, teacher_net, testloader, device, fosc_cal, EPSILON, args.output_image)
    elif args.eval_only:
        args.loss = ''
        test(0, None)
    else:
        main()


