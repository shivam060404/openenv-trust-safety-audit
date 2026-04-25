"""
integration/defender_api.py
============================
COLISEUM Defender — FastAPI Inference Server

Shivam's environment calls this to get BLOCK/ALLOW decisions.
Runs independently of the GRPO training loop.

Usage:
    # Start server (on hackathon day, after Notebook 2 SFT completes)
    python integration/defender_api.py --model_repo okaditya08/coliseum-defender-sft
    
    # After GRPO completes (Saturday PM), restart with GRPO checkpoint:
    python integration/defender_api.py --model_repo okaditya08/coliseum-defender-grpo

API Endpoints (consumed by Shivam's mesa_env.py):
    POST /defender/predict
        Body:  { "prompt": str, "session_id": str }
        Resp:  { "decision": "BLOCK"|"ALLOW", "confidence": float,
                 "reason": str, "latency_ms": float }

    GET  /defender/health
        Resp:  { "status": "ok", "model": str, "device": str }

    GET  /defender/metrics
        Resp:  { "total_calls": int, "block_rate": float,
                 "avg_latency_ms": float }
"""

import os
import re
import time
import argparse
import logging
from pathlib import Path
from typing import Optional
from collections import deque

# Load .env from project root before anything else reads env vars
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent.parent / '.env'
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv optional; env vars must be set manually

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent to path for reward + config modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from reward.mesa_reward import parse_defender_output, compute_reward
from configs.config import (
    SFT_MODEL_REPO,
    SYSTEM_PROMPT,
    INFERENCE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('defender_api')


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# Prefer local path if download_models.py has been run; fall back to HF Hub
_PROJECT_ROOT   = Path(__file__).parent.parent.parent
_LOCAL_ADAPTER  = _PROJECT_ROOT / 'models' / 'adapters' / 'coliseum-defender-sft'
_LOCAL_BASE     = _PROJECT_ROOT / 'models' / 'base'     / 'Qwen2.5-1.5B-Instruct'
DEFAULT_MODEL   = str(_LOCAL_ADAPTER) if _LOCAL_ADAPTER.is_dir() else SFT_MODEL_REPO

MAX_NEW_TOKENS  = INFERENCE['max_new_tokens']
TEMPERATURE     = INFERENCE['temperature']
MAX_INPUT_LEN   = INFERENCE['max_input_len']


# ─────────────────────────────────────────────────────────────────────────────
# Global model state (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

_model     = None
_tokenizer = None
_model_id  = None
_device    = None

# Metrics ring buffer
_call_log = deque(maxlen=500)

# Per-session memory: session_id -> list of past records (newest at end)
# Each record: { turn_index, prompt, decision, confidence, reason, true_label }
_session_memory: dict = {}
_session_counters: dict = {}
SESSION_MEMORY_MAX = 20


def load_model(model_repo: str, hf_token: Optional[str] = None):
    """Load defender model. Called once at startup. Prefers local disk, falls back to HF Hub."""
    global _model, _tokenizer, _model_id, _device

    src = "local disk" if os.path.isdir(model_repo) else "HF Hub"
    logger.info(f'Loading model from {src}: {model_repo}')

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    dtype  = torch.float16 if device in ('cuda', 'mps') else torch.float32

    try:
        # Try Unsloth first (CUDA only)
        if torch.cuda.is_available():
            from unsloth import FastLanguageModel
            _model, _tokenizer = FastLanguageModel.from_pretrained(
                model_name     = model_repo,
                max_seq_length = 512,
                dtype          = None,
                load_in_4bit   = True,
                token          = hf_token,
            )
            FastLanguageModel.for_inference(_model)
            logger.info('✅ Loaded via Unsloth (CUDA fast inference)')
        else:
            raise ImportError("No CUDA — use PEFT path")

    except (ImportError, Exception) as e:
        logger.info(f'Unsloth skipped ({type(e).__name__}), using transformers+PEFT on {device}')
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        # Use local base if available, else download
        base_id = str(_LOCAL_BASE) if _LOCAL_BASE.is_dir() else 'Qwen/Qwen2.5-1.5B-Instruct'
        logger.info(f'  Base model : {base_id}')
        logger.info(f'  Adapter    : {model_repo}')

        _tokenizer = AutoTokenizer.from_pretrained(model_repo, token=hf_token)
        base       = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=dtype, device_map={'': device}, token=hf_token
        )
        _model = PeftModel.from_pretrained(base, model_repo, token=hf_token).eval()
        logger.info(f'✅ Loaded via PEFT on {device}')

    _model_id = model_repo
    _device   = str(next(_model.parameters()).device)
    logger.info(f'   Device: {_device}')
    logger.info(f'   VRAM:   {torch.cuda.memory_allocated()/1e9:.2f}GB used')


def build_audit_message(prompt_text: str, memory: Optional[list] = None) -> str:
    """Compose the user message, optionally injecting per-session memory of the
    defender's own past decisions and the reward each decision earned. The
    defender does NOT see ground-truth labels directly — only the reward signal
    (and its TP/FP/FN/TN bucket) so it can adapt to its own past errors."""
    head = f'Audit this prompt:\n\n{prompt_text[:MAX_INPUT_LEN]}'
    if not memory:
        return head

    lines = ['## Your recent decisions in this session and the reward each earned:']
    for r in memory:
        decided = r['decision']
        conf    = r.get('confidence', 0.0)
        score   = r.get('reward_score')
        label   = r.get('reward_label')
        if score is None:
            tail = '(reward pending)'
        else:
            tail = f'reward={score:+.2f} ({label})'
        snippet = (r.get('prompt') or '')[:140].replace('\n', ' ')
        lines.append(f'  - "{snippet}" -> {decided} (conf={conf:.2f}), {tail}')
    lines.append('')
    lines.append('Use this self-feedback to improve your next call. A negative reward')
    lines.append('on a past turn means that decision was wrong; adjust accordingly.')
    lines.append('')
    lines.append(head)
    return '\n'.join(lines)


@torch.no_grad()
def run_defender(prompt_text: str, memory: Optional[list] = None) -> dict:
    """
    Run the defender model on a prompt.
    Returns parsed decision dict.
    Called per-request by the API.
    """
    user_content = build_audit_message(prompt_text, memory)
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': user_content}
    ]

    # apply_chat_template as string first, then tokenize separately so we get
    # a proper BatchEncoding (input_ids + attention_mask) for generate()
    text   = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenizer(text, return_tensors='pt').to(_model.device)

    t0 = time.perf_counter()

    output_ids = _model.generate(
        **inputs,
        max_new_tokens = MAX_NEW_TOKENS,
        temperature    = TEMPERATURE,
        do_sample      = True,
        pad_token_id   = _tokenizer.eos_token_id,
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    new_tokens  = output_ids[0][inputs['input_ids'].shape[-1]:]
    raw_output  = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    parsed      = parse_defender_output(raw_output)

    return {
        **parsed,
        'latency_ms': round(latency_ms, 1),
        'raw_output': raw_output,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = 'COLISEUM Defender API',
    description = 'Trust & Safety defender model inference endpoint',
    version     = '1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ['*'],
    allow_methods  = ['*'],
    allow_headers  = ['*'],
)


class PredictRequest(BaseModel):
    prompt:        str
    session_id:    str  = 'default'
    use_memory:    bool = False    # opt-in for back-compat; True for stateful sessions
    memory_window: int  = 5        # how many past records to inject


class PredictResponse(BaseModel):
    decision:   str    # 'BLOCK' | 'ALLOW'
    confidence: float
    reason:     str
    latency_ms: float
    session_id: str
    format_ok:  bool
    turn_index: int    # monotonic per session — pass to /defender/feedback


class RewardRequest(BaseModel):
    session_id:  str
    turn_index:  int
    true_label:  str    # 'UNSAFE' | 'SAFE' — used ONLY to compute the reward
    attack_tier: int = 1  # 1=DAN, 2=Wild, 3=Mutation — for tier bonus


class HealthResponse(BaseModel):
    status:     str
    model:      str
    device:     str
    vram_gb:    float


@app.get('/defender/health', response_model=HealthResponse)
def health():
    """Shivam pings this to confirm defender is running before starting GRPO."""
    if _model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return HealthResponse(
        status  = 'ok',
        model   = _model_id or 'unknown',
        device  = _device or 'unknown',
        vram_gb = round(vram, 2),
    )


@app.post('/defender/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Main endpoint — called by Shivam's mesa_env.py per turn.
    
    Shivam's environment calls this AFTER the attacker generates a prompt,
    to get the defender's BLOCK/ALLOW decision.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet')

    if not req.prompt or len(req.prompt.strip()) < 3:
        raise HTTPException(status_code=422, detail='Prompt too short')

    # Build per-session memory window (server-side state)
    memory_slice: list = []
    if req.use_memory:
        full_mem = _session_memory.get(req.session_id, [])
        w = req.memory_window if req.memory_window and req.memory_window > 0 else len(full_mem)
        memory_slice = full_mem[-w:]

    result = run_defender(req.prompt, memory_slice)

    # Allocate monotonic turn_index and append session record
    turn_idx = _session_counters.get(req.session_id, 0)
    _session_counters[req.session_id] = turn_idx + 1
    record = {
        'turn_index':   turn_idx,
        'prompt':       req.prompt,
        'decision':     result['decision'],
        'confidence':   result['confidence'],
        'reason':       result['reason'],
        'reward_score': None,   # filled in by /defender/record_reward
        'reward_label': None,   # TP / TN / FP / FN
        'ts':           time.time(),
    }
    sess_mem = _session_memory.setdefault(req.session_id, [])
    sess_mem.append(record)
    if len(sess_mem) > SESSION_MEMORY_MAX:
        del sess_mem[0:len(sess_mem) - SESSION_MEMORY_MAX]

    # Log for metrics
    _call_log.append({
        'session_id': req.session_id,
        'decision':   result['decision'],
        'confidence': result['confidence'],
        'latency_ms': result['latency_ms'],
        'ts':         time.time(),
    })

    logger.info(
        f'[{req.session_id[:8]}#{turn_idx}] {result["decision"]} '
        f'conf={result["confidence"]:.2f} '
        f'lat={result["latency_ms"]:.0f}ms '
        f'mem={len(memory_slice)}'
    )

    return PredictResponse(
        decision   = result['decision'],
        confidence = result['confidence'],
        reason     = result['reason'],
        latency_ms = result['latency_ms'],
        session_id = req.session_id,
        format_ok  = result['format_ok'],
        turn_index = turn_idx,
    )


@app.post('/defender/record_reward')
def record_reward(req: RewardRequest):
    """Compute and stamp the reward for a past prediction onto its session
    memory record. The defender's next call will see the reward (and TP/FP/FN/TN
    label) but NOT the raw ground-truth label — adaptation happens through the
    reward signal, mirroring how GRPO trains it offline."""
    mem = _session_memory.get(req.session_id)
    if not mem:
        raise HTTPException(status_code=404, detail=f'unknown session {req.session_id}')
    label = req.true_label.upper()
    if label not in ('UNSAFE', 'SAFE'):
        raise HTTPException(status_code=422, detail='true_label must be UNSAFE or SAFE')
    for r in mem:
        if r['turn_index'] == req.turn_index:
            reward = compute_reward(
                decision    = r['decision'],
                confidence  = r['confidence'],
                true_label  = label,
                attack_tier = req.attack_tier,
                format_ok   = True,
            )
            r['reward_score'] = reward['score']
            r['reward_label'] = reward['label']
            return {
                'status':       'ok',
                'turn_index':   req.turn_index,
                'reward_score': reward['score'],
                'reward_label': reward['label'],
                'breakdown':    reward['breakdown'],
            }
    raise HTTPException(status_code=404, detail=f'turn_index {req.turn_index} not in session')


@app.post('/defender/session/reset')
def reset_session(session_id: str = 'default'):
    """Clear per-session memory and turn counter."""
    _session_memory.pop(session_id, None)
    _session_counters.pop(session_id, None)
    return {'status': 'ok', 'session_id': session_id}


@app.get('/defender/session/{session_id}')
def get_session(session_id: str):
    """Inspect server-side memory for a session."""
    mem = _session_memory.get(session_id, [])
    return {
        'session_id':  session_id,
        'turn_count':  _session_counters.get(session_id, 0),
        'memory_size': len(mem),
        'memory':      mem,
    }


@app.get('/defender/metrics')
def metrics():
    """Live metrics for Shivam's dashboard and pitch demo."""
    if not _call_log:
        return {'total_calls': 0, 'block_rate': 0.0, 'avg_latency_ms': 0.0}

    total      = len(_call_log)
    blocks     = sum(1 for c in _call_log if c['decision'] == 'BLOCK')
    avg_lat    = sum(c['latency_ms'] for c in _call_log) / total
    avg_conf   = sum(c['confidence'] for c in _call_log) / total

    return {
        'total_calls':    total,
        'block_rate':     round(blocks / total, 3),
        'allow_rate':     round(1 - blocks / total, 3),
        'avg_latency_ms': round(avg_lat, 1),
        'avg_confidence': round(avg_conf, 3),
        'model':          _model_id,
    }


@app.post('/defender/reload')
def reload_model(model_repo: str, hf_token: str = ''):
    """
    Hot-reload endpoint — called by Aditya on Saturday PM
    to swap SFT checkpoint for GRPO checkpoint without restarting.
    """
    try:
        load_model(model_repo, hf_token or os.environ.get('HF_TOKEN'))
        return {'status': 'ok', 'model': model_repo}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_repo', default=DEFAULT_MODEL)
    parser.add_argument('--host',       default='0.0.0.0')
    parser.add_argument('--port',       type=int, default=8001)
    parser.add_argument('--hf_token',   default=os.environ.get('HF_TOKEN', ''))
    args = parser.parse_args()

    # Load model before starting server
    load_model(args.model_repo, args.hf_token)

    logger.info(f'🚀 Starting Defender API at http://{args.host}:{args.port}')
    logger.info(f'   Health:         GET  /defender/health')
    logger.info(f'   Predict:        POST /defender/predict        (use_memory=True for stateful sessions)')
    logger.info(f'   Record reward:  POST /defender/record_reward  (stamps reward onto past decision)')
    logger.info(f'   Session:        GET  /defender/session/{{id}}   POST /defender/session/reset')
    logger.info(f'   Metrics:        GET  /defender/metrics')
    logger.info(f'   Reload:         POST /defender/reload')

    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
