#!/usr/bin/python3
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from agents.question_model_RN import QAgent

# -----------------------------
# CONFIG
# -----------------------------
TOPICS_4 = [
    "Syllogisms",
    "Seating Arrangements (Linear, Circular)",
    "Family tree logic",
    "Mixed Series (Alphanumeric)",
]

TARGET_PER_TOPIC = 100
TOTAL_BUDGET_TOKENS = 1024
CONTENT_BUDGET_TOKENS = 150

# How many candidates to generate per attempt (per topic item)
K_CANDIDATES = 3

# Safety cap to avoid infinite loops
MAX_ATTEMPTS_PER_TOPIC = 1200

# -----------------------------
# Helpers
# -----------------------------
def _strip_choice_prefix(s: str) -> str:
    s = "" if s is None else str(s)
    return re.sub(r"^[A-Da-d]\)\s*", "", s).strip()

def _norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def signature(obj: Dict[str, Any]) -> str:
    topic = _norm(obj.get("topic", ""))
    question = _norm(obj.get("question", ""))
    choices = obj.get("choices", [])
    if not isinstance(choices, list):
        choices = []
    ch = [_norm(_strip_choice_prefix(c)) for c in choices[:4]]
    return topic + "||" + question + "||" + "||".join(ch)

def has_valid_answer(obj: Dict[str, Any]) -> bool:
    ans = str(obj.get("answer", "")).strip().upper()
    return ans in {"A", "B", "C", "D"}

def has_4_choices(obj: Dict[str, Any]) -> bool:
    ch = obj.get("choices", [])
    if not isinstance(ch, list) or len(ch) != 4:
        return False
    # must start with A/B/C/D
    for i, c in enumerate(ch):
        if not isinstance(c, str) or len(c) < 2:
            return False
        if c[0].upper() not in "ABCD":
            return False
    return True

def parse_first_json(s: str) -> Optional[Dict[str, Any]]:
    if not isinstance(s, str):
        return None
    m = re.search(r"\{.*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            if "answer" not in obj and "expected_answer" in obj:
                obj["answer"] = obj["expected_answer"]
            return obj
        return None
    except Exception:
        return None

def content_token_count(agent: QAgent, obj: Dict[str, Any]) -> int:
    topic = str(obj.get("topic", ""))
    question = str(obj.get("question", ""))
    answer = str(obj.get("answer", ""))
    choices = obj.get("choices", [])
    if not isinstance(choices, list):
        choices = []
    txt = " ".join([topic, question, answer] + [str(c) for c in choices[:4]])
    return len(agent.tokenizer.encode(txt, add_special_tokens=False))

def trim_explanation_to_budget(agent: QAgent, obj: Dict[str, Any]) -> None:
    exp = str(obj.get("explanation", ""))
    ctk = content_token_count(agent, obj)
    remaining = max(0, TOTAL_BUDGET_TOKENS - ctk)
    tok = agent.tokenizer
    ids = tok.encode(exp, add_special_tokens=False)
    if len(ids) <= remaining:
        return
    obj["explanation"] = tok.decode(ids[:remaining], skip_special_tokens=True).strip()

def score_candidate(agent: QAgent, obj: Dict[str, Any], seen_sigs: set) -> float:
    """
    Local reward function (no extra LLM calls).
    Higher is better.
    """
    score = 0.0

    # format correctness
    if isinstance(obj, dict):
        score += 1.0
    else:
        return -999.0

    # required keys
    for k in ["topic", "question", "choices", "answer", "explanation"]:
        if k in obj:
            score += 0.5
        else:
            score -= 2.0

    if has_4_choices(obj):
        score += 3.0
    else:
        score -= 6.0

    if has_valid_answer(obj):
        score += 3.0
    else:
        score -= 8.0

    qtxt = str(obj.get("question", "")).strip()
    if qtxt.endswith("?"):
        score += 1.0
    else:
        score -= 1.0

    # content token budget
    ctk = content_token_count(agent, obj)
    if ctk <= CONTENT_BUDGET_TOKENS:
        score += 4.0
        # prefer closer to, but under, budget (often means richer but still compliant)
        score += min(1.0, ctk / CONTENT_BUDGET_TOKENS)
    else:
        score -= 10.0

    # discourage duplicates
    sig = signature(obj)
    if sig in seen_sigs:
        score -= 12.0
    else:
        score += 1.5

    # mild "difficulty" heuristics
    # (not perfect, but helps)
    if any(w in qtxt.lower() for w in ["exactly", "at least", "at most", "immediately", "opposite", "third", "second"]):
        score += 0.7

    exp = str(obj.get("explanation", "")).strip()
    if len(exp) >= 20:
        score += 0.7
    if len(exp) == 0:
        score -= 2.0

    return score

def build_prompt_for_topic(topic: str) -> Tuple[str, str]:
    """
    Minimal prompt: strong JSON + constraints + token budget.
    """
    sys_prompt = (
        "You are an expert-level examiner.\n"
        "Output ONLY one valid JSON object.\n"
        "English ONLY.\n"
        "No newlines anywhere.\n"
        "All keys must be present: topic, question, choices, answer, explanation.\n"
        "choices must be exactly 4 strings: A) ... B) ... C) ... D) ...\n"
        "answer must be exactly one of: A,B,C,D.\n"
        "No extra text outside JSON.\n"
    )

    user_prompt = (
        f"Generate ONE extremely difficult puzzle-based MCQ on topic: {topic}.\n"
        "Hard constraints:\n"
        "1) JSON only, no markdown, no extra text.\n"
        "2) No newline characters anywhere.\n"
        "3) Exactly 4 choices: A) ... B) ... C) ... D) ...\n"
        "4) Exactly one correct answer; answer is one of A/B/C/D.\n"
        "5) Keep (topic+question+choices+answer) under 150 tokens.\n"
        "6) explanation can be longer but within total 1024 tokens.\n"
        "Return JSON like:\n"
        "{\"topic\":\"...\",\"question\":\"...?\",\"choices\":[\"A) ...\",\"B) ...\",\"C) ...\",\"D) ...\"],\"answer\":\"A\",\"explanation\":\"...\"}"
    )
    return user_prompt, sys_prompt

def generate_k(agent: QAgent, topic: str, k: int) -> List[str]:
    prompt, sys_prompt = build_prompt_for_topic(topic)
    # generate K by passing list of same prompt (small randomness via sampling)
    prompts = [prompt] * k
    resp, _, _ = agent.generate_response(prompts, sys_prompt, tgps_show=False)
    if isinstance(resp, list):
        return resp
    return [resp]

# -----------------------------
# MAIN
# -----------------------------
def main(out_dir: str = "outputs"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qagent = QAgent()

    # final accepted rows
    final_rows: List[Dict[str, Any]] = []
    seen_sigs = set()

    # DPO preference pairs: jsonl with {"prompt":..., "chosen":..., "rejected":...}
    pref_lines: List[Dict[str, str]] = []

    per_topic_counts = {t: 0 for t in TOPICS_4}
    attempts = {t: 0 for t in TOPICS_4}

    for topic in TOPICS_4:
        while per_topic_counts[topic] < TARGET_PER_TOPIC and attempts[topic] < MAX_ATTEMPTS_PER_TOPIC:
            attempts[topic] += 1

            # generate K candidates
            raw_list = generate_k(qagent, topic, K_CANDIDATES)

            # parse + score
            parsed: List[Dict[str, Any]] = []
            scored: List[Tuple[float, Dict[str, Any], str]] = []

            prompt, _ = build_prompt_for_topic(topic)

            for raw in raw_list:
                obj = None
                try:
                    obj = json.loads(raw)
                    if "answer" not in obj and "expected_answer" in obj:
                        obj["answer"] = obj["expected_answer"]
                except Exception:
                    obj = parse_first_json(raw)

                if not isinstance(obj, dict):
                    continue

                # normalize topic field to requested topic family
                obj["topic"] = topic

                # trim explanation locally (won't fix content>150; that’s rejected)
                trim_explanation_to_budget(qagent, obj)

                sc = score_candidate(qagent, obj, seen_sigs)
                scored.append((sc, obj, raw))
                parsed.append(obj)

            if not scored:
                continue

            scored.sort(key=lambda x: x[0], reverse=True)
            best_sc, best_obj, best_raw = scored[0]
            worst_sc, worst_obj, worst_raw = scored[-1]

            # record preference pair (for DPO) ONLY if best is meaningfully better
            # (avoids noisy pairs)
            if best_sc - worst_sc >= 4.0:
                pref_lines.append({
                    "prompt": prompt,
                    "chosen": json.dumps(best_obj, ensure_ascii=False),
                    "rejected": json.dumps(worst_obj, ensure_ascii=False),
                })

            # accept only if valid format + within budgets + not duplicate
            if best_sc < 0:
                continue
            if not has_4_choices(best_obj):
                continue
            if not has_valid_answer(best_obj):
                continue
            if content_token_count(qagent, best_obj) > CONTENT_BUDGET_TOKENS:
                continue
            sig = signature(best_obj)
            if sig in seen_sigs:
                continue

            seen_sigs.add(sig)
            final_rows.append(best_obj)
            per_topic_counts[topic] += 1

            if per_topic_counts[topic] % 10 == 0:
                print(f"[{topic}] collected {per_topic_counts[topic]}/{TARGET_PER_TOPIC} (attempts {attempts[topic]})", flush=True)

        print(f"DONE {topic}: {per_topic_counts[topic]}/{TARGET_PER_TOPIC} after {attempts[topic]} attempts", flush=True)

    # Save outputs
    train_path = out_dir / "train_400.json"
    prefs_path = out_dir / "prefs_400.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(final_rows, f, indent=2, ensure_ascii=False)

    with open(prefs_path, "w", encoding="utf-8") as f:
        for line in pref_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nSaved:\n- {train_path}\n- {prefs_path}\nTotal rows: {len(final_rows)}\nTotal prefs: {len(pref_lines)}")

if __name__ == "__main__":
    main()
