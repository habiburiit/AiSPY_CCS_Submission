#!/bin/bash
# =============================================================================
# AiSPY Sabotage Attack — Final Experimental Commands

# =============================================================================

# GPU 1 — CIFAR-10: ResNet-18 (alpha=50, tau=0.55) + ResNet-50 (alpha=100, tau=0.55) + VGG-16 (alpha=100, tau=0.90)
CUDA_VISIBLE_DEVICES=1 nohup bash -c '
python Gradient_Based_Sabotage_Attack.py \
    --dataset cifar10 --model resnet18 --data_dir ./data \
    --epochs 120 --lr 0.1 --lr_milestones 72 102 --lr_gamma 0.1 \
    --detection grad_norm --norm_threshold 0.55 --alpha 50.0 \
    --outdir ./results_gradnorm_cifar10_resnet18 | tee gn_cifar10_resnet18.log && \
python Gradient_Based_Sabotage_Attack.py \
    --dataset cifar10 --model resnet50 --data_dir ./data \
    --epochs 120 --lr 0.1 --lr_milestones 72 102 --lr_gamma 0.1 \
    --detection grad_norm --norm_threshold 0.55 --alpha 100.0 \
    --outdir ./results_gradnorm_cifar10_resnet50 | tee gn_cifar10_resnet50.log && \
python Gradient_Based_Sabotage_Attack.py \
    --dataset cifar10 --model vgg16 --data_dir ./data \
    --epochs 120 --lr 0.1 --lr_milestones 72 102 --lr_gamma 0.1 \
    --detection grad_norm --norm_threshold 0.90 --alpha 100.0 --batch_size 64 \
    --outdir ./results_gradnorm_cifar10_vgg16 | tee gn_cifar10_vgg16.log
' > gpu1_sabotage.log 2>&1 &

# GPU 2 — CIFAR-100: ResNet-18 (alpha=50, tau=0.55) + ResNet-50 (alpha=100, tau=0.55) + VGG-16 (alpha=100, tau=0.90)
CUDA_VISIBLE_DEVICES=2 nohup bash -c '
python Gradient_Based_Sabotage_Attack.py \
    --dataset cifar100 --model resnet18 --data_dir ./data \
    --epochs 200 --lr 0.1 --lr_milestones 120 170 --lr_gamma 0.1 \
    --detection grad_norm --norm_threshold 0.55 --alpha 50.0 \
    --outdir ./results_gradnorm_cifar100_resnet18 | tee gn_cifar100_resnet18.log && \
python Gradient_Based_Sabotage_Attack.py \
    --dataset cifar100 --model resnet50 --data_dir ./data \
    --epochs 200 --lr 0.1 --lr_milestones 120 170 --lr_gamma 0.1 \
    --detection grad_norm --norm_threshold 0.55 --alpha 100.0 \
    --outdir ./results_gradnorm_cifar100_resnet50 | tee gn_cifar100_resnet50.log && \
python Gradient_Based_Sabotage_Attack.py \
    --dataset cifar100 --model vgg16 --data_dir ./data \
    --epochs 200 --lr 0.1 --lr_milestones 120 170 --lr_gamma 0.1 \
    --detection grad_norm --norm_threshold 0.90 --alpha 100.0 --batch_size 64 \
    --outdir ./results_gradnorm_cifar100_vgg16 | tee gn_cifar100_vgg16.log
' > gpu2_sabotage.log 2>&1 &

# GPU 3 — ImageNet: ResNet-18 (alpha=100, tau=0.85) + ResNet-50 (alpha=50, tau=0.55) + VGG-16 (loss_plateau, alpha=100)
CUDA_VISIBLE_DEVICES=3 nohup bash -c '
python Gradient_Based_Sabotage_Attack.py \
    --dataset imagenet --model resnet18 --data_dir ./imagenet-val \
    --use_val_as_train --val_split_ratio 0.8 \
    --epochs 30 --lr 0.01 --lr_milestones 15 25 --lr_gamma 0.1 \
    --detection grad_norm --norm_threshold 0.85 --alpha 100.0 \
    --outdir ./results_gradnorm_imagenet_resnet18 | tee gn_imagenet_resnet18.log && \
python Gradient_Based_Sabotage_Attack.py \
    --dataset imagenet --model resnet50 --data_dir ./imagenet-val \
    --use_val_as_train --val_split_ratio 0.8 \
    --epochs 30 --lr 0.01 --lr_milestones 15 25 --lr_gamma 0.1 \
    --detection grad_norm --norm_threshold 0.55 --alpha 50.0 \
    --outdir ./results_gradnorm_imagenet_resnet50 | tee gn_imagenet_resnet50.log && \
python Gradient_Based_Sabotage_Attack.py \
    --dataset imagenet --model vgg16 --data_dir ./imagenet-val \
    --use_val_as_train --val_split_ratio 0.8 \
    --epochs 30 --lr 0.01 --lr_milestones 15 25 --lr_gamma 0.1 \
    --detection loss_plateau --conv_threshold 0.20 --window_N 3 \
    --alpha 100.0 --batch_size 128 \
    --outdir ./results_gradnorm_imagenet_vgg16 | tee gn_imagenet_vgg16.log
' > gpu3_sabotage.log 2>&1 &

echo "All 9 jobs launched across GPUs 1, 2, 3."
echo "Monitor with: tail -f gpu1_sabotage.log gpu2_sabotage.log gpu3_sabotage.log"
