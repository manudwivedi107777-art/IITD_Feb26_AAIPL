#!/usr/bin/python3

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


def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = s.replace("```json", "").replace("```", "").strip()
    s = re.sub(r"<think>[\s\S]*?</think>", "", s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


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

        # Auto-load adapter (env var first, else default path)
        adapter_path = os.getenv("Q_ADAPTER_PATH", "outputs/qwen14b-qagent-lora").strip()
        default_adapter = Path("outputs/qwen14b-qagent-lora")

        if not adapter_path and default_adapter.exists():
            adapter_path = str(default_adapter)

        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    def generate_response(
        self,
        message: Union[str, List[str]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
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

        max_new = int(kwargs.get("max_new_tokens", 1024))
        max_new = min(max_new, 1024)

        generated_kwargs = dict(
            max_new_tokens=max_new,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=kwargs.get("do_sample", True),
            temperature=kwargs.get("temperature", 0.2),
            top_p=kwargs.get("top_p", 0.9),
            repetition_penalty=kwargs.get("repetition_penalty", 1.15),
        )

        start = time.time() if tgps_show else None
        generated = self.model.generate(**model_inputs, **generated_kwargs)
        gen_time = (time.time() - start) if tgps_show else None

        outs: List[str] = []
        token_len = 0

        for in_ids, out_ids in zip(model_inputs.input_ids, generated):
            new_tokens = out_ids[len(in_ids) :].tolist()
            if tgps_show:
                token_len += len(new_tokens)

            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            outs.append(_clean_text(decoded))

        resp = outs[0] if is_single else outs
        if tgps_show:
            return resp, token_len, gen_time
        return resp, None, None

    def _build_user_prompt(self, topic: str) -> str:
        return (
            f"Topic: {topic}\n"
            "Generate ONE high-quality puzzle-based MCQ strictly as JSON:\n"
            "{\"topic\":\"...\",\"question\":\"...?\",\"choices\":[\"A) ...\",\"B) ...\",\"C) ...\",\"D) ...\"],\"answer\":\"A|B|C|D\",\"explanation\":\"...\"}\n"
            "Rules: English only, no newlines, exactly 4 choices, exactly one correct answer."
        )

    def _example_to_sft_text(self, ex: Dict[str, Any]) -> str:
        topic = str(ex.get("topic", "")).replace("\n", " ").strip()
        question = str(ex.get("question", "")).replace("\n", " ").strip()
        choices = [str(c).replace("\n", " ").strip() for c in ex.get("choices", [])]

        # ✅ FIX: accept expected_answer also (so training never sees empty answer)
        answer = str(ex.get("answer", ex.get("expected_answer", ""))).strip()

        explanation = str(ex.get("explanation", "")).replace("\n", " ").strip()

        user_prompt = self._build_user_prompt(topic)

        assistant_json = {
            "topic": topic,
            "question": question,
            "choices": choices,
            "answer": answer,
            "explanation": explanation,
        }
        assistant_text = json.dumps(assistant_json, ensure_ascii=False)

        msgs = [
            {"role": "system", "content": "You are a strict JSON generator. Output ONLY JSON. English only. No newlines."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
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
        with open(data_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        # ✅ FIX: normalize rows so 'expected_answer' becomes 'answer' (training-safe)
        for ex in rows:
            if isinstance(ex, dict) and ("answer" not in ex) and ("expected_answer" in ex):
                ex["answer"] = ex["expected_answer"]

        texts = [self._example_to_sft_text(ex) for ex in rows]
        ds = Dataset.from_dict({"text": texts})

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

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

    if args.train:
        QAgent().train_lora(
            data_path=args.data,
            output_dir=args.out,
            epochs=args.epochs,
            batch_size=args.batch,
            grad_accum=args.grad_accum,
            lr=args.lr,
        )
