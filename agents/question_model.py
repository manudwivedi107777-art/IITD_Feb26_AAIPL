#!/usr/bin/python3
# agents/question_model_RN.py

import os
import re
import time
import json
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


# ============================================================
# Utility: Clean output for JSON safety
# ============================================================

def _clean_text_for_json(s: str) -> str:
    """Light cleanup to reduce JSON breakage (keeps content, removes obvious wrappers)."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # remove code fences
    s = s.replace("```json", "").replace("```", "").strip()
    # remove think blocks (some models emit these even when disabled)
    s = re.sub(r"<think>[\s\S]*?</think>", "", s)
    # normalize whitespace to single spaces
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _extract_first_json_object(s: str) -> Optional[str]:
    """Extract the first {...} JSON object using brace matching."""
    if not s:
        return None
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _validate_mcq_obj(obj: Dict[str, Any]) -> bool:
    """Schema + basic sanity checks."""
    required = ["topic", "question", "choices", "answer", "explanation"]
    if not isinstance(obj, dict):
        return False
    if not all(k in obj for k in required):
        return False

    if not isinstance(obj["topic"], str) or not obj["topic"].strip():
        return False
    if not isinstance(obj["question"], str) or not obj["question"].strip():
        return False
    if not isinstance(obj["choices"], list) or len(obj["choices"]) != 4:
        return False
    if not all(isinstance(c, str) and c.strip() for c in obj["choices"]):
        return False
    if not isinstance(obj["answer"], str) or obj["answer"] not in ["A", "B", "C", "D"]:
        return False
    if not isinstance(obj["explanation"], str) or not obj["explanation"].strip():
        return False
    return True


def _build_fix_prompt(bad_text: str) -> str:
    return (
        "Fix the following into ONE valid minified JSON object ONLY (no extra text, no markdown).\n"
        "Schema:\n"
        "{\"topic\":\"...\",\"question\":\"...\",\"choices\":[\"A) ...\",\"B) ...\",\"C) ...\",\"D) ...\"],"
        "\"answer\":\"A|B|C|D\",\"explanation\":\"...\"}\n"
        "Rules:\n"
        "- Output ONLY JSON\n"
        "- English only\n"
        "- No newline characters\n"
        "- Exactly one correct answer\n"
        f"BAD_OUTPUT: {bad_text}"
    )


# ============================================================
# QAgent
# ============================================================

class QAgent(object):
    def __init__(self, **kwargs):
        self.base_model_path = str(
            Path(__file__).parent.parent
            / "hf_models"
            / "models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, padding_side="left"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype="auto",
            device_map="auto",
        )

        adapter_path = os.getenv("Q_ADAPTER_PATH", "").strip()
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    # ============================================================
    # Inference
    # ============================================================

    def generate_response(
        self,
        message: Union[str, List[str]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        Returns:
          - if tgps_show=True: (resp, token_len, gen_time)
          - else: resp
        resp:
          - dict for single prompt when JSON parses + validates
          - list of dicts for multi prompt when JSON parses + validates
          - if parsing fails even after a retry, falls back to cleaned string(s)
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        tgps_show = bool(kwargs.get("tgps_show", False))
        is_single = isinstance(message, str)
        messages = [message] if is_single else message

        texts: List[str] = []
        for msg in messages:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            text = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            texts.append(text)

        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=kwargs.get("max_new_tokens", 1024),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=kwargs.get("do_sample", True),
            temperature=kwargs.get("temperature", 0.45),
            top_p=kwargs.get("top_p", 0.9),
            repetition_penalty=kwargs.get("repetition_penalty", 1.12),
        )

        start = time.time() if tgps_show else None
        generated = self.model.generate(**model_inputs, **gen_kwargs)
        gen_time = (time.time() - start) if tgps_show else None

        outs: List[Union[Dict[str, Any], str]] = []
        token_len = 0

        # 1) decode + clean + try parse/validate
        for in_ids, out_ids in zip(model_inputs.input_ids, generated):
            new_tokens = out_ids[len(in_ids) :].tolist()
            if tgps_show:
                token_len += len(new_tokens)

            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            decoded = _clean_text_for_json(decoded)

            json_str = _extract_first_json_object(decoded)
            parsed: Optional[Dict[str, Any]] = None
            if json_str:
                try:
                    parsed = json.loads(json_str)
                except Exception:
                    parsed = None

            if parsed is not None and not _validate_mcq_obj(parsed):
                parsed = None

            outs.append(parsed if parsed is not None else decoded)

        # 2) one retry pass for any failures (strings)
        need_retry = [i for i, o in enumerate(outs) if isinstance(o, str)]
        if need_retry:
            retry_texts: List[str] = []
            for i in need_retry:
                fix_msg = _build_fix_prompt(str(outs[i]))
                chat = [
                    {"role": "system", "content": "You output ONLY JSON."},
                    {"role": "user", "content": fix_msg},
                ]
                retry_texts.append(
                    self.tokenizer.apply_chat_template(
                        chat,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )

            retry_inputs = self.tokenizer(
                retry_texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            retry_gen = self.model.generate(**retry_inputs, **gen_kwargs)

            for local_idx, (in_ids, out_ids) in enumerate(
                zip(retry_inputs.input_ids, retry_gen)
            ):
                new_tokens2 = out_ids[len(in_ids) :].tolist()
                if tgps_show:
                    token_len += len(new_tokens2)

                decoded2 = self.tokenizer.decode(new_tokens2, skip_special_tokens=True)
                decoded2 = _clean_text_for_json(decoded2)

                json_str2 = _extract_first_json_object(decoded2)
                parsed2: Optional[Dict[str, Any]] = None
                if json_str2:
                    try:
                        parsed2 = json.loads(json_str2)
                    except Exception:
                        parsed2 = None

                if parsed2 is not None and _validate_mcq_obj(parsed2):
                    outs[need_retry[local_idx]] = parsed2
                else:
                    # keep cleaned string fallback
                    outs[need_retry[local_idx]] = decoded2

        resp = outs[0] if is_single else outs

        if tgps_show:
            return resp, token_len, gen_time
        return resp

    # ============================================================
    # Training (LoRA)
    # ============================================================

    def _build_prompt(self, topic: str) -> str:
        return (
            f"Given topic: {topic}\n"
            "Generate ONE puzzle-based MCQ strictly in this JSON format:\n"
            "{"
            "\"topic\":\"...\","
            "\"question\":\"...\","
            "\"choices\":[\"A) ...\",\"B) ...\",\"C) ...\",\"D) ...\"],"
            "\"answer\":\"A|B|C|D\","
            "\"explanation\":\"...\""
            "}\n"
            "Rules:\n"
            "- Output ONLY JSON\n"
            "- English only\n"
            "- No newline characters\n"
            "- Exactly one correct answer\n"
        )

    def _example_to_sft_text(self, ex: Dict[str, Any]) -> str:
        topic = str(ex.get("topic", "")).replace("\n", " ").strip()
        question = str(ex.get("question", "")).replace("\n", " ").strip()
        choices = [str(c).replace("\n", " ").strip() for c in ex.get("choices", [])]
        answer = str(ex.get("answer", "")).strip()
        explanation = str(ex.get("explanation", "")).replace("\n", " ").strip()

        user_prompt = self._build_prompt(topic)

        assistant_json = {
            "topic": topic,
            "question": question,
            "choices": choices,
            "answer": answer,
            "explanation": explanation,
        }

        assistant_text = json.dumps(assistant_json, ensure_ascii=False)

        messages = [
            {"role": "system", "content": "You are a precise JSON generator."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def train_lora(
        self,
        data_path: str,
        output_dir: str,
        epochs: int = 2,
        batch_size: int = 1,
        grad_accum: int = 8,
        lr: float = 2e-4,
    ):
        # Load data
        with open(data_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        # Build dataset (avoid heavy .map)
        texts = [self._example_to_sft_text(ex) for ex in rows]
        ds = Dataset.from_dict({"text": texts})

        # Fresh model load for training
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # LoRA config
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj", "gate_proj"
            ],
        )

        model = get_peft_model(model, lora_cfg)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
            bf16=True,
            fp16=False,
            optim="adamw_torch",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=ds,
            args=training_args,
        )

        trainer.train()

        trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"LoRA adapter saved to: {output_dir}")


# ============================================================
# CLI for training
# ============================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data", type=str)
    ap.add_argument("--out", type=str)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    agent = QAgent()

    if args.train:
        agent.train_lora(
            data_path=args.data,
            output_dir=args.out,
            epochs=args.epochs,
            batch_size=args.batch,
            grad_accum=args.grad_accum,
            lr=args.lr,
        )
