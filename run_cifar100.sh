#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/huggingface
export TOKENIZERS_PARALLELISM=true

# for model in WideResNet ResNet18 MobileNetV2
teacher_path=/root/checkpoint/tinyproject/CIFAR10/TRADES-CIFAR10-ResNet18-lambd1.0/optimal_epoch93_ckpt.t7

for teacher_model in ResNet18 
do 
    for dataset in CIFAR100
    do
        for model in MobileNetV2
        do
            name=distill-T-${teacher_model}-S-${model}-D-${dataset}

            python -u main_d.py --teacher_model ${teacher_model} \
            --model ${model} \
            --output ${name} \
            --loss KD \
            --dataset ${dataset} \
            --teacher_path ${teacher_path} \
            --debug | tee log/${name}.log
        done
    done
done
