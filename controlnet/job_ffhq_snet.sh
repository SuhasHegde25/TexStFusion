#! /bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --time=10:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

export http_proxy=http://proxy-10g.10g.siddhi.param:9090
export https_proxy=http://proxy-10g.10g.siddhi.param:9090

accelerate launch train_controlnet_ffhq.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --dataset_name Ryan-sjtu/ffhq512-caption --output_dir /nlsasfs/home/obfuscated/ghsuhas/codebase/subjectnet_v/trained_models/ffhq/subjectnet_onemod --num_train_epochs 3 --checkpointing_steps 30000 --resolution 512 --train_batch_size 64  --use_8bit_adam --enable_xformers_memory_efficient_attention --set_grads_to_none --learning_rate 1e-5 --train_vit_controlnet