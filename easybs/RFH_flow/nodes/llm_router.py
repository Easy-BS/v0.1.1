# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 08:28:09 2025

@author: Xiguan Liang @SKKU
"""

# ./RFH_flow/nodes/llm_router.py


from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict

from state_schema import SimulationState

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

SYSTEM_PROMPT = """You are an AI agent for building simulation.
Your job: read the user's text and output STRICT JSON only (no markdown).

Schema:
{
  "intent": "add_rfh" | "ask_clarification" | "unknown",
  "rfh_targets": string[],
  "clarification_question": string
}

Rules:
- If user asks to add/install radiant floor heating in specific rooms -> intent="add_rfh" and list rfh_targets.
- If rooms are missing or ambiguous -> intent="ask_clarification" and write a short question asking for room names exactly as in the building.
- If unrelated -> intent="unknown".
Return JSON only.
"""

def _openai_chat_json(user_text: str, model: str) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    }

    req = urllib.request.Request(
        OPENAI_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)

    content = data["choices"][0]["message"]["content"]
    # Must be JSON only
    return json.loads(content)

def llm_router(state: SimulationState) -> SimulationState:
    text = (state.get("user_input") or "").strip()
    if not text:
        return {"errors": ["No user input received."]}

    model = state.get("model") or "gpt-4o-mini"

    try:
        agent = _openai_chat_json(text, model=model)
    except Exception as e:
        return {"errors": [f"LLM router failed: {e}"]}

    intent = agent.get("intent", "unknown")
    out: SimulationState = {"agent_json": agent, "intent": intent}

    if intent == "add_rfh":
        targets = agent.get("rfh_targets") or []
        targets = [str(x).strip() for x in targets if str(x).strip()]
        if not targets:
            out["intent"] = "ask_clarification"
            out["clarification_question"] = "Which rooms should receive RFH? Please list them (e.g., \"Room_1\", \"Living_2\")."
        else:
            out["rfh_targets"] = targets

    elif intent == "ask_clarification":
        out["clarification_question"] = agent.get("clarification_question") or \
            "Which rooms should receive RFH? Please list them (e.g., \"Room_1\", \"Living_2\")."

    return out
