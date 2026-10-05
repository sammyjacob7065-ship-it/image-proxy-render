import os
import re
import time
import uuid
import random
import shutil
import threading
import mimetypes
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from gradio_client import Client

HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY", "")
SPACE_TIMEOUT = int(os.getenv("SPACE_TIMEOUT", "90"))     # max seconds per Space
COOLDOWN = int(os.getenv("COOLDOWN", "600"))              # skip a failed Space for 10 min
JOB_TTL = 3600                                            # delete jobs/images after 1 hour

WORK_DIR = Path("/tmp/images")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Ordered by how likely they are to be fast and reliable.
# Dead or renamed Spaces are skipped automatically; check /spaces/check to see which are alive.
SPACES = [
    "black-forest-labs/FLUX.1-schnell",
    "Tongyi-MAI/Z-Image-Turbo",
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-3.5-large-turbo",
    "ByteDance/SDXL-Lightning",
    "ByteDance/Hyper-SDXL-1Step-T2I",
    "multimodalart/FLUX.1-merged",
    "DamarJati/FLUX.1-RealismLora",
    "prithivMLmods/FLUX-LoRA-DLC",
    "KingNish/Realtime-FLUX",
    "stabilityai/stable-diffusion-3.5-large",
    "playgroundai/playground-v2.5",
]

app = FastAPI(title="Image Proxy")
jobs: dict = {}
clients: dict = {}
plans: dict = {}        # space -> (api_name, parameters) discovered once
bad_until: dict = {}    # space -> timestamp until which it is skipped
lock = threading.Lock()

PROMPT_RE = re.compile(r"^(prompt|text|text_prompt|query|caption|input_text)$", re.I)
NEG_RE = re.compile(r"neg", re.I)
IMAGE_COMPONENTS = {"image", "gallery"}


class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""   # accepted so old n8n bodies still work


def check_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def get_client(space: str) -> Client:
    with lock:
        c = clients.get(space)
    if c is None:
        c = Client(space, token=HF_TOKEN) if HF_TOKEN else Client(space)
        with lock:
            clients[space] = c
    return c


def discover(space: str):
    """Find the endpoint that takes a prompt and returns an image."""
    if space in plans:
        return plans[space]
    client = get_client(space)
    info = client.view_api(return_format="dict", print_info=False)
    endpoints = info.get("named_endpoints", {}) or {}
    best = None
    for api_name, ep in endpoints.items():
        params = ep.get("parameters", []) or []
        returns = ep.get("returns", []) or []
        has_prompt = any(PROMPT_RE.match(str(p.get("parameter_name") or p.get("label") or "")) for p in params)
        returns_image = any(str(r.get("component", "")).lower() in IMAGE_COMPONENTS for r in returns)
        if has_prompt and returns_image:
            best = (api_name, params)
            break
    if not best:
        raise RuntimeError("no text-to-image endpoint found")
    plans[space] = best
    return best


def build_kwargs(params: list, prompt: str, negative: str) -> dict:
    kwargs = {}
    for p in params:
        name = str(p.get("parameter_name") or p.get("label") or "")
        low = name.lower()
        has_default = p.get("parameter_has_default", False)
        if PROMPT_RE.match(name):
            kwargs[name] = prompt
        elif NEG_RE.search(low):
            kwargs[name] = negative
        elif low == "randomize_seed":
            kwargs[name] = True
        elif low == "seed":
            kwargs[name] = random.randint(0, 2**31 - 1)
        elif has_default:
            continue                      # let the Space use its own default
        elif low in ("width", "height"):
            kwargs[name] = 1024
        elif "step" in low:
            kwargs[name] = 4
        elif "guidance" in low or "cfg" in low:
            kwargs[name] = 3.5
        else:
            raise RuntimeError(f"required argument '{name}' not supported")
    return kwargs


def find_path(obj) -> Optional[str]:
    if isinstance(obj, str):
        return obj if os.path.isfile(obj) else None
    if isinstance(obj, dict):
        for key in ("path", "image", "value", "url"):
            if key in obj:
                p = find_path(obj[key])
                if p:
                    return p
        for v in obj.values():
            p = find_path(v)
            if p:
                return p
    if isinstance(obj, (list, tuple)):
        for v in obj:
            p = find_path(v)
            if p:
                return p
    return None


def generate_with_space(space: str, prompt: str, negative: str) -> str:
    api_name, params = discover(space)
    client = get_client(space)
    job = client.submit(**build_kwargs(params, prompt, negative), api_name=api_name)
    try:
        result = job.result(timeout=SPACE_TIMEOUT)
    except Exception:
        try:
            job.cancel()
        except Exception:
            pass
        raise
    src = find_path(result)
    if not src:
        raise RuntimeError(f"no image file in result: {str(result)[:150]}")
    return src


def cleanup():
    now = time.time()
    for jid in [j for j, v in list(jobs.items()) if now - v.get("created", now) > JOB_TTL]:
        for f in WORK_DIR.glob(f"{jid}*"):
            f.unlink(missing_ok=True)
        jobs.pop(jid, None)


def run_job(job_id: str, prompt: str, negative: str):
    jobs[job_id]["status"] = "processing"
    errors = {}
    now = time.time()
    order = [s for s in SPACES if bad_until.get(s, 0) <= now] or list(SPACES)
    for space in order:
        try:
            src = generate_with_space(space, prompt, negative)
            dest = WORK_DIR / f"{job_id}{Path(src).suffix or '.png'}"
            shutil.copy(src, dest)
            jobs[job_id].update(status="done", image_path=str(dest), space_used=space)
            bad_until.pop(space, None)
            return
        except Exception as e:
            errors[space] = str(e)[:200]
            bad_until[space] = time.time() + COOLDOWN
            with lock:
                clients.pop(space, None)
            plans.pop(space, None)
    jobs[job_id].update(status="failed", errors=errors)


@app.get("/")
def home():
    return {"status": "Image Proxy is running", "spaces_configured": len(SPACES)}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/spaces/check")
def spaces_check(x_api_key: Optional[str] = Header(default=None)):
    """Tests which Spaces are alive and understood (does not generate images)."""
    check_key(x_api_key)

    def probe(space):
        try:
            api_name, params = discover(space)
            return space, {"ok": True, "api_name": api_name,
                           "args": [p.get("parameter_name") for p in params]}
        except Exception as e:
            return space, {"ok": False, "error": str(e)[:200]}

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = dict(ex.map(probe, SPACES))
    return {"alive": sum(1 for r in results.values() if r["ok"]), "spaces": results}


@app.post("/generate")
def generate(req: PromptRequest, x_api_key: Optional[str] = Header(default=None)):
    check_key(x_api_key)
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is empty")
    cleanup()
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "created": time.time()}
    threading.Thread(target=run_job, args=(job_id, req.prompt, req.negative_prompt), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str, request: Request, x_api_key: Optional[str] = Header(default=None)):
    check_key(x_api_key)
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    out = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "done":
        out["space_used"] = job["space_used"]
        out["image_url"] = str(request.base_url) + f"image/{job_id}"
    if job["status"] == "failed":
        out["errors"] = job.get("errors")
    return out


@app.get("/image/{job_id}")
def image(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Not ready")
    path = job["image_path"]
    return FileResponse(path, media_type=mimetypes.guess_type(path)[0] or "image/png")
