from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client
import httpx
import random

app = FastAPI()

# Primary + Fallbacks
SPACES = [
    "black-forest-labs/FLUX.1-schnell",          # Primary
    "DamarJati/FLUX.1-RealismLora",
    "prithivMLmods/FLUX-LoRA-DLC",
    "prithivMLmods/FLUX-REALISM",
    "black-forest-labs/FLUX.1-dev",
    "multimodalart/FLUX.1-dev",
    "John6666/flux-lora-the-explorer",
    "strangerzonehf/Flux-Super-Realism-LoRA",
    "hugovntr/flux-schnell-realism",
    "XLabs-AI/flux-RealismLora",
    "togethercomputer/FLUX-schnell-free"
    "DamarJati/FLUX.1-RealismLora",
]

class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""

@app.get("/")
def home():
    return {"status": "Image Proxy is running"}

@app.post("/generate")
async def generate_image(req: PromptRequest):
    last_error = None

    for space in SPACES:
        try:
            client = Client(space)
            # Most Flux spaces use /predict or /generate
            result = client.predict(
                req.prompt,
                api_name="/predict"   # try /predict first
            )
            # result is usually a path or url
            return {
                "success": True,
                "space_used": space,
                "image": result
            }
        except Exception as e:
            last_error = str(e)
            continue

    raise HTTPException(status_code=500, detail=f"All spaces failed. Last error: {last_error}")
