#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/huggingface
export TOKENIZERS_PARALLELISM=true

# for model in WideResNet ResNet18 MobileNetV2
model_path=/root/checkpoint/tinyproject/CIFAR10/TRADES-CIFAR10-ResNet18-lambd1.0/optimal_epoch93_ckpt.t7

loss=TRADES
project_name=pretrain
for lambd in 1
do
for dataset in CIFAR10 CIFAR100
do
    for model in MobileNetV2
    do
        name=pretrain-${loss}-lambd${lambd}-${dataset}-${model}

        python -u main_d.py \
        --project_name ${project_name} \
        --model ${model} \
        --output ${name} \
        --loss ${loss} \
        --dataset ${dataset} \
        --lambd ${lambd} \
        --resume_epoch 0 \
        --epochs 100 \
        | tee log/${name}.log
    done
done
done
