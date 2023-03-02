#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6

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
for aux_alpha in MOST LEAST TargetSE SE
# for aux_alpha in MOST
do 
    # for aux_loss in SAT KL SE
    for aux_loss in SE
    do

        name=${loss}-forget-${aux_loss}-${aux_alpha}
        project_name=COMB2

        python -u main_d.py --teacher_model ${teacher_model} \
        --model ${model} \
        --output ${name} \
        --loss ${loss} \
        --dataset ${dataset} \
        --teacher_path ${teacher_path} \
        --project_name ${project_name} \
        --resume_epoch 70 \
        --epochs 110 \
        | tee log/${name}.log
    done
done
