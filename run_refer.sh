model_path=/root/checkpoint/distill_project/distill-T-ResNet18-S-MobileNetV2-D-CIFAR10-ARD/epoch69/model_ckpt.t7

python main_d.py --eval_only --model MobileNetV2 \
--model_path ${model_path} \
--dataset CIFAR10