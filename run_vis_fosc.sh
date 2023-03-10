#!/bin/bash
# Mar 8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6

# for model in WideResNet ResNet18 MobileNetV2
teacher_model=ResNet18
model=MobileNetV2
teacher_path=/root/checkpoint/CT/CIFAR10/COMB-memorization-SAT-FOSC-aux_lamda_1/epoch99/teacher_ckpt.t7
model_path=/root/checkpoint/CT/CIFAR10/COMB-memorization-SAT-FOSC-aux_lamda_1/epoch99/model_ckpt.t7
loss=COMB
dataset=CIFAR10
memorization=1
aux_lamda=1
teacher_lr=1e-5
project_name=Batchmean
name=vis
image=fig/99.png


python -u main_d.py --teacher_model ${teacher_model} \
--model ${model} \
--output ${name} \
--loss ${loss} \
--dataset ${dataset} \
--model_path ${model_path} \
--teacher_path ${teacher_path} \
--project_name ${project_name} \
--resume_epoch 70 \
--epochs 100 \
--teacher_lr ${teacher_lr} \
--output_image ${image} \
--memorization ${memorization} 
