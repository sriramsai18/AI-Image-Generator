from diffusers import StableDiffusionPipeline
import torch
from  PIL import Image
from IPython.display import display

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt =" a field with a tree in middle."
image = pipe(prompt).images[0]
image.save("generated_img.png")
display(image)