from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
_CACHE_DIR = Path(
    os.environ.get("MEPR_HF_CACHE_DIR", Path(__file__).resolve().parents[2] / "hf_cache")
)

_TOKENIZER = None
_MODEL = None
_MODEL_ID = None


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_id: str = DEFAULT_MODEL_ID):
    global _TOKENIZER, _MODEL, _MODEL_ID
    if _TOKENIZER is not None and _MODEL is not None and _MODEL_ID == model_id:
        return _TOKENIZER, _MODEL

    _TOKENIZER = AutoTokenizer.from_pretrained(model_id, cache_dir=_CACHE_DIR)
    dtype = torch.float16 if _get_device() == "cuda" else torch.float32
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=_CACHE_DIR,
    )
    _MODEL.to(_get_device())
    _MODEL.eval()
    _MODEL_ID = model_id
    return _TOKENIZER, _MODEL


def _parse_questions(text: str, max_items: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                items = [str(x).strip() for x in data if str(x).strip()]
                return items[:max_items]
        except Exception:
            pass

    lines = []
    for raw in text.splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[\).]|[-*]+)\s*", "", raw).strip()
        if cleaned:
            lines.append(cleaned)
    return lines[:max_items]


def generate_questions(
    job_title: str,
    job_description: str,
    num_questions: int = 5,
    model_id: str = DEFAULT_MODEL_ID,
) -> List[str]:
    tokenizer, model = _load_model(model_id=model_id)

    prompt = (
        "You are a hiring interviewer. Based on the job title and description, "
        f"generate {num_questions} interview questions. "
        "Focus on behavioral and skills-based questions. "
        "Write in English. Return ONLY a JSON array of strings.\n\n"
        f"Job title: {job_title}\n"
        f"Job description: {job_description}"
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return _parse_questions(content, max_items=int(num_questions))


def generate_explanation(
    summary: str,
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    tokenizer, model = _load_model(model_id=model_id)

    prompt = (
        "You are an HR career consultant. Write a concise, clear explanation in English. "
        "Explain whether the candidate fits the role, why, and what to improve. "
        "Mention emotional-state caution if present. Offer 3 alternative roles from the list. "
        "If the requested role is not found, explicitly say you used the closest match. "
        "Use a friendly, professional tone. Do not mention that you are an AI.\n\n"
        f"{summary}"
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content.strip()


def unload_model() -> None:
    global _TOKENIZER, _MODEL, _MODEL_ID
    _TOKENIZER = None
    _MODEL_ID = None
    if _MODEL is not None:
        try:
            del _MODEL
        except Exception:
            pass
    _MODEL = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
