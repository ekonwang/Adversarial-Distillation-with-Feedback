#!/bin/bash
# edit on Mar 8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/huggingface
export TOKENIZERS_PARALLELISM=true

teacher_path=/root/checkpoint/tinyproject/CIFAR10/TRADES-CIFAR10-ResNet18-lambd1.0/optimal_epoch93_ckpt.t7
model_path=/root/checkpoint/distill_project/distill-T-ResNet18-S-MobileNetV2-D-CIFAR10-ARD/epoch69/model_ckpt.t7
memorization=0

loss=KL-Coarse
project_name=Batchmean

for memorization in 0 1
do
for teacher_model in ResNet18 
do 
    for dataset in CIFAR10
    do
        for model in MobileNetV2
        do
            if [ ${memorization} -gt 0 ] 
            then
                name=kl-coarse-memorization
            else
                name=kl-coarse
            fi

            python -u main_d.py --teacher_model ${teacher_model} \
            --project_name ${project_name} \
            --model ${model} \
            --output ${name} \
            --loss ${loss} \
            --dataset ${dataset} \
            --teacher_path ${teacher_path} \
            --model_path ${model_path} \
            --resume_epoch 70 \
            --memorization ${memorization} \
             | tee log/${name}.log
        done
    done
done
done