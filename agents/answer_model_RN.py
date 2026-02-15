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


class AAgent(object):
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
        
        # --------------------------------------------------
        # Adapter Loading Logic (priority-based)
        # --------------------------------------------------
        
        # 1) explicit argument override (if passed via AnsweringAgent kwargs)
        adapter_path = kwargs.get("adapter_path", None)
        
        # 2) environment variable fallback
        if not adapter_path:
            adapter_path = os.getenv("A_ADAPTER_PATH", "").strip()
        
        # 3) default local path fallback
        if not adapter_path:
            default_path = Path("outputs/aagent-lora")
            if default_path.exists():
                adapter_path = str(default_path)
        
        # 4) load if found
        if adapter_path:
            from peft import PeftModel
            print(f"[A-Agent] Loading LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        else:
            print("[A-Agent] No adapter found. Using base model.")


        self.model.eval()

    # --------------------------------------------------
    # INFERENCE
    # --------------------------------------------------
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

        max_new = int(kwargs.get("max_new_tokens", 512))
        max_new = min(max_new, 512)

        gen_kwargs = dict(
            max_new_tokens=max_new,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=kwargs.get("do_sample", True),
            temperature=kwargs.get("temperature", 0.2),
            top_p=kwargs.get("top_p", 0.9),
            repetition_penalty=kwargs.get("repetition_penalty", 1.2),
        )

        start = time.time() if tgps_show else None
        generated = self.model.generate(**model_inputs, **gen_kwargs)
        gen_time = (time.time() - start) if tgps_show else None

        outs: List[str] = []
        token_len = 0

        for in_ids, out_ids in zip(model_inputs.input_ids, generated):
            new_tokens = out_ids[len(in_ids) :].tolist()
            if tgps_show:
                token_len += len(new_tokens)

            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            decoded = _clean_text(decoded)
            outs.append(decoded)

        resp = outs[0] if is_single else outs
        if tgps_show:
            return resp, token_len, gen_time
        return resp, None, None

    # --------------------------------------------------
    # LORA TRAINING FOR ANSWERING
    # --------------------------------------------------
    def train_lora(
        self,
        data_path: str,
        output_dir: str,
        epochs: int = 2,
        batch_size: int = 1,
        grad_accum: int = 8,
        lr: float = 2e-4,
    ):
        """
        data_path must be filtered_questions.json
        containing:
            question
            choices
            answer
        """

        with open(data_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        formatted_texts = []

        for ex in rows:
            question = ex["question"]
            choices = ex["choices"]
            correct_answer = ex["answer"].strip().upper()

            formatted_choices = []
            for idx, choice in enumerate(choices[:4]):
                letter = "ABCD"[idx]
                formatted_choices.append(f"{letter}) {choice.strip()}")

            user_prompt = (
                "INSTRUCTIONS:\n"
                "1) Exactly one option is correct.\n"
                "2) Return ONLY valid JSON.\n"
                "3) answer must be one letter: A/B/C/D.\n"
                "4) reasoning <= 100 words.\n\n"
                f"Question: {question}\n"
                f"Choices: {' '.join(formatted_choices)}\n\n"
                "Return JSON:\n"
                '{"answer":"A|B|C|D","reasoning":"..."}'
            )

            assistant_output = json.dumps(
                {
                    "answer": correct_answer,
                    "reasoning": "The selected option correctly satisfies the conditions given in the question."
                },
                ensure_ascii=False
            )

            chat = [
                {"role": "system", "content": "You are an expert quantitative aptitude solver."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_output},
            ]

            text = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )

            formatted_texts.append(text)

        ds = Dataset.from_dict({"text": formatted_texts})

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


# --------------------------------------------------
# CLI TRAINING ENTRY
# --------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train A-Agent with LoRA")

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to filtered_questions.json",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save LoRA adapter",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device batch size",
    )

    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )

    args = parser.parse_args()

    agent = AAgent()

    agent.train_lora(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
    )
