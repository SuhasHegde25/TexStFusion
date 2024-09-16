from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import AutoImageProcessor, ViTModel
from diffusers.utils import load_image
import torch
import random
import argparse

import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(
        description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')

parser.add_argument('--model_version', type=str, help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--controlnet', type=str, help='Path to deploy.prototxt', default='hed_edge_detector/deploy.prototxt', required=False)
parser.add_argument('--subjectnet', type=str, help='Path to hed_pretrained_bsds.caffemodel', default='hed_edge_detector/hed_pretrained_bsds.caffemodel', required=False)
parser.add_argument('--samples', type=int, help='Path to hed_pretrained_bsds.caffemodel', default='hed_edge_detector/hed_pretrained_bsds.caffemodel', required=False)

args = parser.parse_args()

base_model_path = "runwayml/stable-diffusion-v1-5"
vit_model_path = "google/vit-base-patch16-224-in21k"

controlnet_path = args.controlnet
subjectnet_path = args.subjectnet


#folder_prefix = "./inference/" + vit_controlnet_path + '_' + controlnet_path

multi_controlnet = True

canny_as_control = True

num_samples = args.samples

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=True)
nets = controlnet
if multi_controlnet:
    subjectnet = ControlNetModel.from_pretrained(subjectnet_path, torch_dtype=torch.float16, use_safetensors=True, is_subjectnet=True)
    nets = [controlnet,subjectnet]

#vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=nets, torch_dtype=torch.float16, use_safetensors=True, 
    safety_checker = None,
    requires_safety_checker = False
)
#image_processor = AutoImageProcessor.from_pretrained(vit_model_path)
#vit_model = ViTModel.from_pretrained(vit_model_path)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

#network_weightage = [1.0,0.0]
#network_weightages = [[0.1,0.9],[0.2,0.8],[1.0,0.0],[0.0,1.0],[0.3,0.7],[0.4,0.6],[0.5,0.5],[0.6,0.4],[0.7,0.3],[0.5,0.3]]
network_weightages = [[0.5,0.3],[1.0,0.0]]

input_folder = "/nlsasfs/home/obfuscated/ghsuhas/codebase/subjectnet_v/inferences/inputs/church/"

output_folder_prefix = "/nlsasfs/home/obfuscated/ghsuhas/codebase/subjectnet_v/inferences/" + args.model_version + "/"

#subject_image_names = ["vit_control.jpg", "vit_control_test.jpg"] #, "vit_control_test2.jpg", "black.jpg"]
#subject_image_names = ["vit_control_test3.jpg","vit_control_test4.jpg"]
#subject_image_names = ["vit_control.jpg", "vit_control_test.jpg","vit_control_test3.jpg","vit_control_test4.jpg"]
#subject_image_names = ["809.jpg", "vit_cat_control.jpg"]
#subject_image_names = ["dog1.jpg", "dog2.jpg","dog3.jpg","dog4.jpg"]
#subject_image_names = ["df1.jpg", "df2.jpg","df3.jpg","df4.jpg","df31.jpg"]
#subject_image_names = ["port4.jpg", "port13.jpg","port19.jpg","port27.jpg"]
#subject_image_names = ["4b.jpg", "6b.jpg","7b.jpg","16b.jpg","19b.jpg"]
#subject_image_names = ["dog1.jpg", "dog2.jpg","dog3.jpg"]#,"dog5.jpg", "dog6.jpg"]
#subject_image_names = ["port7.jpg", "port8.jpg"]#,"port9.jpg","port10.jpg", "port11.jpg", "port12.jpg"]
subject_image_names = ["church1.jpg", "church2.jpg","church3.jpg","church4.jpg", "church5.jpg", "church6.jpg", "church7.jpg"]
#subject_image_names = ["./inference/inputs/pose_subject3.jpg", "./inference/inputs/pose_subject2.jpg", "./inference/inputs/black.jpg"]
#vit_image_names = ["./809.jpg", "./black.jpg"]
#vit_image_names = ["./celeb_rm.jpg", "./black.jpg"]
#vit_image_names = ["./vit_cat_control_rm.jpg", "./black.jpg"]
#vit_image_names = ["./vit_tiger_control_2.png", "./black.jpg"]
#vit_image_names = ["./vit_lion.png", "./black.jpg"]
#vit_image_names = ["./vit_mili_rm.jpg", "./black.jpg"]
#subject_image_names = ["./inference/inputs/ac_rm.jpg", "./inference/inputs/celeb_rm.jpg", "./inference/inputs/black.jpg"]
#subject_image_names = ["./inference/inputs/vit_control_ac2.jpg", "./inference/inputs/vit_control_test.jpg", "./inference/inputs/black.jpg"]
#vit_image_names = ["./vit_tiger_control_2.png", "./black.jpg"]
#vit_image = load_image("./black.jpg")
#control_image_names = ["celeb_control.jpg", "celeb_control_test.jpg"] #, "celeb_control_test2.jpg", "celeb_control.jpg"]
#control_image_names = ["celeb_control_test3.jpg","celeb_control_test4.jpg"]
#control_image_names = ["celeb_control.jpg", "celeb_control_test.jpg","celeb_control_test3.jpg","celeb_control_test4.jpg"]
#control_image_names = ["hed_dog1.jpg", "hed_dog2.jpg","hed_dog3.jpg"]#,"hed_dog5.jpg", "hed_dog6.jpg"]
#control_image_names = subject_image_names
#control_image_names = ["kp_port7.jpg", "kp_port8.jpg"]#,"kp_port9.jpg","kp_port10.jpg", "kp_port11.jpg", "kp_port12.jpg"]
control_image_names = ["hed_church1.jpg", "hed_church2.jpg","hed_church3.jpg","hed_church4.jpg", "hed_church5.jpg", "hed_church6.jpg", "hed_church7.jpg"]
#control_image_names = ["pose1.jpg", "pose2.jpg","pose3.jpg","pose4.jpg","pose31.jpg"]
#control_image_names = ["df_hed1.jpg", "df_hed2.jpg","df_hed3.jpg","df_hed4.jpg","df_hed31.jpg"]
#control_image_names = ["port_hed4.jpg", "port_hed13.jpg","port_hed19.jpg","port_hed27.jpg"]
#control_image_names = ["4b_hed.jpg", "6b_hed.jpg","7b_hed.jpg","16b_hed.jpg","19b_hed.jpg"]
#control_image_names = ["./inference/inputs/pose_control3.jpg", "./inference/inputs/pose_control2.jpg", "./inference/inputs/pose_control2.jpg"]
#control_image_names = ["./cat_control_old.jpg", "./cat_control_old.jpg"]
#control_image_names = ["canny_809.jpg", "cat_control.jpg"]
#control_image_names = ["./canny_celeb_rm.jpg", "./ac_canny.jpg"]
#control_image_names = ["./tiger_control_2.jpg", "./cat_control.jpg"]
#control_image_names = ["./inference/inputs/canny_ac_rm.jpg", "./inference/inputs/canny_celeb_rm.jpg", "./inference/inputs/canny_celeb_rm.jpg"]
#control_image_names = ["./control_lion.png", "./control_lion.png"]
#control_image = load_image("./celeb_control.jpg")
#vit_image_names = ["dog1_rm.jpg", "dog2_rm.jpg","dog3_rm.jpg"]#,"dog5_rm.jpg", "dog6_rm.jpg"]
#vit_image_names = ["bg_port7.jpg", "bg_port8.jpg"]#,"bg_port9.jpg","bg_port10.jpg", "bg_port11.jpg", "bg_port12.jpg"]
vit_image_names = ["church1.jpg", "church2.jpg","church3.jpg","church4.jpg", "church5.jpg", "church6.jpg", "church7.jpg"]
#prompt = "a golden circle with bubbles background"
#prompt = "a professional, high definition, detailed, high quality image of a tiger in a jungle"
#prompt = "a professional, high definition, detailed, high quality image"
#prompt = "a photo of a white cat in a garden with red roses"
#prompt = "a man wearing glasses standing on a road"
#prompt = "a woman wearing a white t - shirt with a logo on the front in front of a white background"
#prompt = "a woman wearing a blue shirt with white collar in front of a white background"
#prompt = "a professional, high definition, detailed, high quality image of a white colored lion" 
#prompt = "a professional, high definition, detailed, high quality image of a lion with thick blackish orange fur"
#prompt = "a professional, high definition, detailed, high quality image of a lion with dense orangish black fur"
#prompt = "a golden sculpture of a man" #"a man in a suit with long hair" #"a man in a suit wearing glasses" #"sks"  #"abcdefg"  #"a man in a suit and tie"  #"a man in a T-shirt"
#prompt = "a photo of a cat wearing sunglasses"
#prompts = ["a woman in a suit","a woman wearing glasses","a woman standing on a red carpet with photographers behind her","a woman standing in the mountain","a woman in rain with reflections on the pavement", "painting of a woman in anthonys van dyck style", "black and white sketch of a woman"]
#prompts = ["a photography of a man with mountain in the background","a photography of a man with glasses","a painting of a man in anthonys van dyck style","a black and white sketch of a man","a photography of a woman with blonde hair","a photography of a man wearing suit","a photography of a man smiling for the camera","a photography of a man in jungle with long hair","a colorful oil painting of a man in a frame"]
#prompts = ["a professional, detailed, high quality image"]
#prompts = ["a photo of a church in snow","a photo of a church during sunset","a photo of a golden church","a photo of a church during the night with beautiful lightings","a photo of a church made of wood","a painting of a building in alexander jansson style"]
#prompts = ["a photo of a dog in a garden","a photo of a dog in a cozy bedroom, 4k realistic","a photo of a dog in a garden, orange leaves on the trees and ground, 4k realistic","a photo of a dog on moon, UE5 rendering, ray tracing","a photo of a white dog in front of a mountain"]
#prompts = ["a photo of a white dog in front of a mountain"]
#prompts = ["a photography of a man with mountain in the background"]
prompts = ["a photo of a church during sunset"]
#subject_image_names = vit_image_names

for prompt in tqdm(prompts):
    if not os.path.exists(output_folder_prefix + prompt):
        os.makedirs(output_folder_prefix + prompt)

    for network_weightage in network_weightages:
        output_folder = output_folder_prefix + prompt + "/" + str(network_weightage[0]) + "_" + str(network_weightage[1])

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for i in range(num_samples):
            # generate image
            #seed = random.randint(0, 15000)
            generator = torch.manual_seed(i)

            for j in range(len(subject_image_names)):
                print("\nGenerating sample : "+str(i+1)+", with seed : "+str(i)+", with subject : "+subject_image_names[j]+"\n")

                control_image = load_image(input_folder + control_image_names[j])
                subject_image = load_image(input_folder + subject_image_names[j])
                vit_image = load_image(input_folder + vit_image_names[j])
                if subject_image.size != (512,512):
                    subject_image = subject_image.resize((512,512))
                if control_image.size != (512,512):
                    control_image = control_image.resize((512,512))
                if vit_image.size != (512,512):
                    vit_image = vit_image.resize((512,512))
        
                if canny_as_control:
                    img = np.array(subject_image)
                    edges = cv2.Canny(img, 100, 150)
                    control_image = Image.fromarray(edges)

                if multi_controlnet:
                    control_images = [control_image,subject_image]
                    image = pipe(prompt, num_inference_steps=20, generator=generator, 
                                 image=control_images, vit_conditioning_image=vit_image, #vit_image,
                                 network_weightage=network_weightage).images[0]
                else:
                    #image = pipe(prompt, num_inference_steps=20, generator=generator, image=control_image, vit_conditioning_image=vit_image).images[0]
                    image = pipe(prompt, num_inference_steps=20, generator=generator, image=control_image, vit_conditioning_image=subject_image).images[0]
    
                #image.save("./infers_vit_comp/celeb_v3_p_"+str(j)+"/celeb_v"+str(j)+"_"+str(i+1)+".png")
                #image.save("./infers_vit_comp/celeb_v3_6e_specs_12"+"/celeb_v"+str(j)+"_"+str(i+1)+".png")
                image.save(output_folder + "/v"+str(j)+"_"+str(i+1)+".jpg")