#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/huggingface
export TOKENIZERS_PARALLELISM=true

# for model in WideResNet ResNet18 MobileNetV2
teacher_path=/root/checkpoint/pretrain/teacher-epoch93.t7
model_path=/root/checkpoint/pretrain/stu-epoch69.t7

loss=ARD
project_name=NEW
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
