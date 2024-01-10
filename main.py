
import base64
import io
from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image
from pydantic import BaseModel


class Item(BaseModel):
    prompt: str
    temperature: float = 1.0
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: Optional[int] = 42

pipe = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.load_lora_weights("lora-android-toy/checkpoint-800", weights="pytorch_lora_weights.safetensors")


def run_model(pipe, params, logger):
    if params.seed is not None and int(params.seed) > 0:
        logger.info("Manual seed")
        generator = torch.Generator("cuda").manual_seed(int(params.seed))
    else:
        logger.info("Random seed")
        generator = torch.Generator("cuda")

    logger.info("Seed: {}".format(params.seed))

    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to("cuda")

    images = pipe(
        params.prompt,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        temperature=params.temperature,
        seed=params.seed,
        generator=generator,
    ).images

    return images


def predict(item, run_id, logger):
    params = Item(**item)
    
    if not params.prompt:
      logger.info("User did not send prompt in request")
      return {"status_code": 422, "description": "Please, specify a prompt"}

    generated_image = run_model(pipe=pipe, params=params, logger=logger)[0]
    buffered = io.BytesIO()
    generated_image.save(buffered, format="PNG")
    generated_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"status_code": 200, "generated_image": generated_image}
