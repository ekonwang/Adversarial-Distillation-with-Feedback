#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/root/huggingface
export TOKENIZERS_PARALLELISM=true

teacher_path=/root/checkpoint/cache/resnet_cifar100_ckpt.t7
model_path=/root/checkpoint/cache/distil_mbnv2_cifar100_ckpt.t7

loss=ARD-PRO
project_name=NEW
memorization=1
for aux_lamda in 0.01 0.1 0.25 0.5 1 10
do
    for teacher_model in ResNet18 
    do 
        for dataset in CIFAR100
        do
            for model in MobileNetV2
            do
                if [ ${memorization} -gt 0 ] 
                then
                    name=coarse-memorization-lamda${aux_lamda}
                else
                    name=coarse-lamda${aux_lamda}
                fi

                python -u main_d.py --teacher_model ${teacher_model} \
                --project_name ${project_name} \
                --model ${model} \
                --output ${name} \
                --loss ${loss} \
                --dataset ${dataset} \
                --teacher_path ${teacher_path} \
                --model_path ${model_path} \
                --aux_lamda ${aux_lamda} \
                --resume_epoch 70 \
                --memorization ${memorization} \
                | tee log/${name}.log
            done
        done
    done
done
