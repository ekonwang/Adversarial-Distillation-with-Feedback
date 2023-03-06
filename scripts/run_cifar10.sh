#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/huggingface
export TOKENIZERS_PARALLELISM=true

# for model in WideResNet ResNet18 MobileNetV2
teacher_path=/root/checkpoint/tinyproject/CIFAR10/adv-ResNet18-CIFAR10/optimal_epoch51_ckpt.t7

loss=ARD-PRO
for teacher_model in ResNet18 
do 
    for dataset in CIFAR10
    do
        for model in MobileNetV2
        do
            name=distill-T-${teacher_model}-S-${model}-D-${dataset}-${loss}

            python -u main_d.py --teacher_model ${teacher_model} \
            --model ${model} \
            --output ${name} \
            --loss ${loss} \
            --dataset ${dataset} \
            --teacher_path ${teacher_path} \
            | tee log/${name}.log
        done
    done
done
