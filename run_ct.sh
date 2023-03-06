#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/huggingface
export TOKENIZERS_PARALLELISM=true

# for model in WideResNet ResNet18 MobileNetV2
teacher_model=ResNet18
model=MobileNetV2
teacher_path=/root/checkpoint/tinyproject/CIFAR10/TRADES-CIFAR10-ResNet18-lambd1.0/optimal_epoch93_ckpt.t7
model_path=/root/checkpoint/distill_project/distill-T-ResNet18-S-MobileNetV2-D-CIFAR10-ARD/epoch69/model_ckpt.t7
loss=COMB
dataset=CIFAR10
memorization=1
teacher_lr=1e-5

for memorization in 1
do
    # for aux_alpha in MOST LEAST TargetSE SE
    for aux_alpha in SE
    do 
        for aux_loss in SAT KL SE
        # for aux_loss in SE
        do

            if [ $memorization -gt 0 ] 
            then
                name=${loss}-memorization-${aux_loss}-${aux_alpha}-alpha
                project_name=CT
            else 
                name=${loss}-forget-${aux_loss}-${aux_alpha}
                project_name=CT
            fi

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
            --resume_epoch 70 \
            --epochs 100 \
            --teacher_lr ${teacher_lr} \
            --memorization ${memorization} \
            | tee log/${name}.log
        done
    done
done
