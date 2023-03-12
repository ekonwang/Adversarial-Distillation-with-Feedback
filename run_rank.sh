#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# for model in WideResNet ResNet18 MobileNetV2
teacher_model=ResNet18
model=MobileNetV2
teacher_path=/root/checkpoint/pretrain/teacher-epoch93.t7
model_path=/root/checkpoint/pretrain/stu-epoch69.t7
loss=COMB
dataset=CIFAR10
memorization=1
aux_lamda=1
teacher_lr=1e-5

# for aux_lamda in 1
for aux_lamda in 0.1 1 3 6 10
do
    # for aux_alpha in MOST LEAST TargetSE SE FOSC
    for aux_alpha in FOSC
    do 
        # for aux_loss in SAT KL SE
        for aux_loss in SAT
        do

            project_name=CT
            name=${loss}-memorization-${aux_loss}-${aux_alpha}-aux_lamda_${aux_lamda}-right-rank

            python -u main_d.py --teacher_model ${teacher_model} \
            --model ${model} \
            --output ${name} \
            --loss ${loss} \
            --dataset ${dataset} \
            --model_path ${model_path} \
            --teacher_path ${teacher_path} \
            --project_name ${project_name} \
            --aux_alpha ${aux_alpha} \
            --aux_loss ${aux_loss} \
            --aux_lamda ${aux_lamda} \
            --resume_epoch 70 \
            --epochs 100 \
            --teacher_lr ${teacher_lr} \
            --memorization ${memorization} \
            | tee log/${name}.log
        done
    done
done
