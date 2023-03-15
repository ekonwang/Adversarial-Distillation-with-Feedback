#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/huggingface
export TOKENIZERS_PARALLELISM=true

# for model in WideResNet ResNet18 MobileNetV2
model_path=/root/checkpoint/tinyproject/CIFAR100/trades-ResNet18-CIFAR100/optimal_epoch92_ckpt.t7

loss=TRADES
project_name=pretrain
for lambd in 0.5 1 2 6 10 
do
for dataset in CIFAR100
do
    for model in ResNet18
    do
        name=pretrain-${loss}-lambd${lambd}-${dataset}-${model}

        python -u main_d.py \
        --project_name ${project_name} \
        --model ${model} \
        --output ${name} \
        --loss ${loss} \
        --dataset ${dataset} \
        --model_path ${model_path} \
        --lambd ${lambd} \
        --resume_epoch 70 \
        --epochs 120 \
        | tee log/${name}.log
    done
done
done
