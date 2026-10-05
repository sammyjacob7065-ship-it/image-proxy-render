from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client

app = FastAPI()

SPACES = [
    "black-forest-labs/FLUX.1-schnell",
    "DamarJati/FLUX.1-RealismLora",
    "prithivMLmods/FLUX-LoRA-DLC",
    "prithivMLmods/FLUX-REALISM",
    "black-forest-labs/FLUX.1-dev",
    "multimodalart/FLUX.1-dev",
    "John6666/flux-lora-the-explorer",
    "strangerzonehf/Flux-Super-Realism-LoRA",
    "hugovntr/flux-schnell-realism",
    "XLabs-AI/flux-RealismLora",
    "togethercomputer/FLUX-schnell-free",
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
            # Call default endpoint, no api_name
            result = client.predict(req.prompt)
            return {
                "success": True,
                "space_used": space,
                "image": result
            }
        except Exception as e:
            last_error = str(e)
            continue

    raise HTTPException(status_code=500, detail=f"All spaces failed. Last error: {last_error}")
