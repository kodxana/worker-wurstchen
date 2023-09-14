'''
Contains the handler function that will be called by the serverless.
'''

import os
import torch
from PIL import Image
import concurrent.futures
from diffusers import AutoPipelineForText2Image, WuerstchenCombinedPipeline
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
from diffusers.utils import load_image

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA


# -------------------------------- Load Models ------------------------------- #
def load_combined_pipeline():
    combined_pipe = AutoPipelineForText2Image.from_pretrained(
        "warp-ai/wuerstchen", 
        torch_dtype=torch.float16
    ).to("cuda")
    return combined_pipe

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_base = executor.submit(load_combined_pipeline)

    base = future_base.result()
# ---------------------------------- Helper ---------------------------------- #
def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        image_url = rp_upload.upload_image(job_id, image_path)
        image_urls.append(image_url)
    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        # "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]



def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation using the new schema
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])

    base.scheduler = make_scheduler(job_input['scheduler'], base.scheduler.config)

    # Initialize the combined pipeline
    combined_pipe = load_combined_pipeline()

    # Generate images using the combined pipeline
    output = combined_pipe(
        prompt=job_input['prompt'],
        negative_prompt=job_input['negative_prompt'],
        height=job_input['height'],
        width=job_input['width'],
        prior_guidance_scale=job_input['prior_guidance_scale'],
        num_images_per_prompt=job_input['num_images_per_prompt'],
        num_inference_steps=job_input['num_inference_steps'],
        generator=generator
    ).images

    image_urls = _save_and_upload_images(output, job['id'])

    return {"image_url": image_urls[0]} if len(image_urls) == 1 else {"images": image_urls}

runpod.serverless.start({"handler": generate_image})
