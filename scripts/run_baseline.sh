#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/huggingface
export TOKENIZERS_PARALLELISM=true

# for model in WideResNet ResNet18 MobileNetV2
# teacher_path=/root/checkpoint/tinyproject/CIFAR10/adv-ResNet18-CIFAR10/optimal_epoch51_ckpt.t7
# model_path=/root/checkpoint/distill_project/CIFAR10/distill-T-ResNet18-S-MobileNetV2-D-CIFAR10-ARD-PRO/epoch66/model_ckpt.t7
teacher_path=/root/checkpoint/tinyproject/CIFAR10/TRADES-CIFAR10-ResNet18-lambd1.0/optimal_epoch93_ckpt.t7
model_path=/root/checkpoint/distill_project/distill-T-ResNet18-S-MobileNetV2-D-CIFAR10-ARD/epoch69/model_ckpt.t7

loss=ARD
project_name=Batchmean
for teacher_model in ResNet18 
do 
    for dataset in CIFAR10
    do
        for model in MobileNetV2
        do
            # name=coarse
            # name=coarse-memorization
            name=baseline
            # name=distill-T-${teacher_model}-S-${model}-D-${dataset}-${loss}

            python -u main_d.py --teacher_model ${teacher_model} \
            --project_name ${project_name} \
            --model ${model} \
            --output ${name} \
            --loss ${loss} \
            --dataset ${dataset} \
            --teacher_path ${teacher_path} \
            --model_path ${model_path} \
            --resume_epoch 70 \
            | tee log/${name}.log
        done
    done
done
