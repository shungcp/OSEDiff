import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
import time
from osediff import OSEDiff_test
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_validation_prompt(args, image, model, device='xla:0'):
    validation_prompt = ""
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = lq.to(dtype = weight_dtype)
    #lq_ram = ram_transforms(lq).to(dtype=weight_dtype)
    #captions = inference(lq_ram, model)
    #validation_prompt = f"{captions[0]}, {args.prompt},"
    validation_prompt = ""
    
    return validation_prompt, lq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='preset/datasets/test_dataset/input', help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, default='preset/datasets/test_dataset/output', help='the directory to save the output')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='sd model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--osediff_path", type=str, default='preset/models/osediff.pkl')
    parser.add_argument('--prompt', type=str, default='', help='user prompts')
    parser.add_argument('--ram_path', type=str, default=None)
    parser.add_argument('--ram_ft_path', type=str, default=None)
    parser.add_argument('--save_prompts', type=bool, default=True)
    # precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False) # merge lora weights before inference
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) 
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 

    args = parser.parse_args()

    # initialize the model
    model = OSEDiff_test(args)
    model.to(dtype=torch.bfloat16)
    # get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    # get ram model
    DAPE = ram(pretrained=args.ram_path,
            pretrained_condition=args.ram_ft_path,
            image_size=384,
            vit='swin_l')
    DAPE.eval()
    DAPE.to("xla:0")

    # weight type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.bfloat16

    # set weight type
    DAPE = DAPE.to(dtype=weight_dtype)
    
    if args.save_prompts:
        txt_path = os.path.join(args.output_dir, 'txt')
        os.makedirs(txt_path, exist_ok=True)
    
    # make the output dir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    batch_size = 2 #shun update 1
    for i in range(0, len(image_names), batch_size):
        batch_images = image_names[i:i + batch_size]
        batch_outputs = []
        batch_pil_images = []
        batch_sizes = []
        batch_resize_flags = []

        # Prepare batch
        time0 = time.time()
        for image_name in batch_images:
            print("=================="*8)
            input_image = Image.open(image_name).convert('RGB')
            ori_width, ori_height = input_image.size
            rscale = args.upscale
            resize_flag = False
            if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
                scale = (args.process_size//rscale)/min(ori_width, ori_height)
                input_image = input_image.resize((int(scale*ori_width), int(scale*ori_height)))
                resize_flag = True
            input_image = input_image.resize((input_image.size[0]*rscale, input_image.size[1]*rscale))

            new_width = input_image.width - input_image.width % 8
            new_height = input_image.height - input_image.height % 8
            input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        
            batch_pil_images.append(input_image)
            batch_sizes.append((ori_width, ori_height))
            batch_resize_flags.append(resize_flag)

            # validation prompt , to be parallelized, since model() only support one lq, give up
            validation_prompt, lq = get_validation_prompt(args, input_image, DAPE)

            if args.save_prompts:
                bname = os.path.basename(image_name)
                txt_save_path = f"{txt_path}/{bname.split('.')[0]}.txt"
                with open(txt_save_path, 'w', encoding='utf-8') as f:
                    f.write(validation_prompt)
                    f.close()
            print(f"process {image_name}, tag: {validation_prompt}".encode('utf-8')[:300])

            # translate the image one by one
            with torch.no_grad():
                lq = lq*2-1
                output_image = model(lq.to("xla:0"), prompt=validation_prompt)
                time01 = time.time()
                print(f"inference time is: {time01 - time0}")
                output_image = output_image * 0.5 + 0.5
                batch_outputs.append(output_image)

        # try 2, Stack outputs into a batch
        with torch.no_grad():
            combined_outputs = torch.cat(batch_outputs, dim=0)
            # Mark step for async transfer
            xm.mark_step()
            time02 = time.time()
            print(f"MARK_STEP time is: {time02 - time01}")
            # Asnyc transfer to CPU with minimal precision needed for PIL
            output_ims = combined_outputs.cpu().to(dtype = torch.float32)
            time03 = time.time()
            #print(f"output total time is: {time.time() - time1}")
            print(f"TO CPU time is: {time03 - time02}")
            # try 1, seems no improvement
            #output_im = (output_image[0] * 0.5 + 0.5).to('cpu', dtype = torch.float32)
            #output_im = output_image[0].to('cpu').to(dtype = torch.float32) * 0.5 + 0.5
            # Process each image result
            for idx, (output_im, input_image, (ori_width, ori_height), resize_flag, image_name) in enumerate(
                zip(output_ims, batch_pil_images, batch_sizes, batch_resize_flags, batch_images)):
                #breakpoint()
                output_pil = transforms.ToPILImage()(output_im)
                if args.align_method == 'adain':
                    output_pil = adain_color_fix(target=output_pil, source=input_image)
                elif args.align_method == 'wavelet':
                    output_pil = wavelet_color_fix(target=output_pil, source=input_image)
                else:
                    pass
                if resize_flag:
                    output_pil.resize((int(args.upscale*ori_width), int(args.upscale*ori_height)))

                bname = os.path.basename(image_name)
                output_pil.save(os.path.join(args.output_dir, bname))

            print(f"Total Batch time is: {time.time() - time0}")
