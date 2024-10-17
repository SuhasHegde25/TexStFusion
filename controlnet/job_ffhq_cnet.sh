#! /bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --gres=gpu:A100-SXM4:8
#SBATCH --time=15:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

export http_proxy=http://proxy-10g.10g.siddhi.param:9090
export https_proxy=http://proxy-10g.10g.siddhi.param:9090

accelerate launch train_controlnet_custom_ffhq.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --dataset_name ffhq --output_dir /nlsasfs/home/obfuscated/ghsuhas/codebase/subjectnet_v/trained_models/ffhq/controlnet_kp_52k --num_train_epochs 3 --checkpointing_steps 30000 --resolution 512 --train_batch_size 4  --use_8bit_adam --enable_xformers_memory_efficient_attention --set_grads_to_none --learning_rate 1e-5 --condition_dataset_path /nlsasfs/home/obfuscated/ghsuhas/codebase/subjectnet_v/database/ffhq_kp --dataset_path /nlsasfs/home/obfuscated/ghsuhas/codebase/subjectnet_v/database/ffhq --vit_dataset_path /nlsasfs/home/obfuscated/ghsuhas/codebase/subjectnet_v/database/ffhq_rmbg

docker run -it --gpus all -v $(pwd):/data suhashegde/diffusion:latest

accelerate launch --gpu_ids 6 diffusers/examples/controlnet/train_controlnet_custom_ffhq.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --dataset_name portraits --output_dir diffusers/trained_models/ffhq/cnet_text_enc_1.5 --num_train_epochs 3 --checkpointing_steps 30000 --resolution 512 --train_batch_size 4  --use_8bit_adam --enable_xformers_memory_efficient_attention --set_grads_to_none --learning_rate 1e-5 --condition_dataset_path diffusers/datasets/ffhq --dataset_path diffusers/datasets/ffhq --vit_dataset_path diffusers/datasets/ffhq

# Dreambooth params
accelerate launch --gpu_ids 6 diffusers/examples/dreambooth/train_dreambooth.py  --with_prior_preservation --prior_loss_weight=1.0 --class_data_dir="dreambooth_inferences" --class_prompt="a photo of a dog" --use_8bit_adam --gradient_checkpointing --enable_xformers_memory_efficient_attention --set_grads_to_none --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --instance_data_dir=inferences/ffhq --output_dir dreambooth_inferences/output --instance_prompt="a photo of sks dog" --resolution=512 --train_batch_size=4 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=400