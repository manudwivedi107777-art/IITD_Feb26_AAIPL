#!/usr/bin/python3
"""
A-Agent (Answering Agent) — SFT LoRA + PPO RL (reward = JSON validity + correct option)

Dependencies (must already be installed in your environment):
  - torch
  - transformers
  - datasets
  - peft
  - trl
"""

import os
import re
import time
import json
import inspect
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel

# TRL imports (version tolerant)
try:
    from trl import SFTTrainer, PPOTrainer, PPOConfig
except Exception as e:
    raise RuntimeError("trl is required but could not be imported.") from e

try:
    # some TRL versions export this at top-level
    from trl import AutoModelForCausalLMWithValueHead
except Exception:
    # fallback for older/newer layouts
    try:
        from trl.models import AutoModelForCausalLMWithValueHead  # type: ignore
    except Exception as e:
        raise RuntimeError("Could not import AutoModelForCausalLMWithValueHead from trl.") from e


# -------------------------------
# Utilities
# -------------------------------
def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = s.replace("```json", "").replace("```", "").strip()
    s = re.sub(r"<think>[\s\S]*?</think>", "", s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _safe_parse_answer_json(text: str) -> Tuple[bool, str, str]:
    """
    Returns: (ok, answer_letter, reasoning)
    """
    try:
        obj = json.loads(_clean_text(text))
        ans = str(obj.get("answer", "")).strip().upper()
        reasoning = str(obj.get("reasoning", "")).strip()
        if ans in {"A", "B", "C", "D"}:
            return True, ans, reasoning
        return False, "", ""
    except Exception:
        return False, "", ""


def _load_rows(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("data_path must be a JSON list of dicts.")
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"Row {i} is not a dict.")
        for k in ("question", "choices", "answer"):
            if k not in r:
                raise ValueError(f"Row {i} missing key: {k}")
    return rows


def _normalize_choices(choices: List[str]) -> List[str]:
    """
    Ensures we output exactly 4 choices as:
      ["A) ...", "B) ...", "C) ...", "D) ..."]
    Accepts inputs either already prefixed or not.
    """
    if not isinstance(choices, list) or len(choices) < 4:
        raise ValueError("choices must be a list with at least 4 items.")
    out = []
    for i in range(4):
        letter = "ABCD"[i]
        ch = str(choices[i]).strip()
        ch = re.sub(r"^[A-Da-d]\)\s*", "", ch).strip()
        out.append(f"{letter}) {ch}")
    return out


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _make_ppo_config(**kwargs) -> PPOConfig:
    # PPOConfig signature differs across TRL versions (e.g., may not accept seed)
    filtered = _filter_kwargs_for_callable(PPOConfig.__init__, {"self": None, **kwargs})
    filtered.pop("self", None)
    return PPOConfig(**filtered)


def _make_ppo_trainer(ppo_config: PPOConfig, **kwargs) -> PPOTrainer:
    """
    PPOTrainer signatures differ across TRL versions:
      - PPOTrainer(config=..., model=..., ref_model=..., tokenizer=...)
      - PPOTrainer(args=..., ...)
      - PPOTrainer(ppo_config, ...)
    We'll adapt automatically.
    """
    init_sig = inspect.signature(PPOTrainer.__init__)
    params = [p for p in init_sig.parameters.keys() if p != "self"]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}

    if "config" in params:
        filtered_kwargs["config"] = ppo_config
        return PPOTrainer(**filtered_kwargs)

    if "args" in params:
        filtered_kwargs["args"] = ppo_config
        return PPOTrainer(**filtered_kwargs)

    if "ppo_config" in params:
        filtered_kwargs["ppo_config"] = ppo_config
        return PPOTrainer(**filtered_kwargs)

    # fallback positional
    return PPOTrainer(ppo_config, **filtered_kwargs)


def _make_sft_trainer(**kwargs):
    """
    SFTTrainer signature differs:
      - may need dataset_text_field
      - may accept tokenizer
    """
    init_sig = inspect.signature(SFTTrainer.__init__)
    params = [p for p in init_sig.parameters.keys() if p != "self"]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
    return SFTTrainer(**filtered_kwargs)


# -------------------------------
# A-Agent Class
# -------------------------------
class AAgent(object):
    def __init__(self, **kwargs):
        # Base model path (your local HF snapshot) — FIXED PATH JOIN
        self.base_model_path = str(
            Path(__file__).parent.parent
            / "hf_models"
            / "models--Qwen--Qwen2.5-14B-Instruct"
            / "snapshots"
            / "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, padding_side="left", trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        # Adapter Loading Logic (priority-based)
        adapter_path = kwargs.get("adapter_path", None)
        if not adapter_path:
            adapter_path = os.getenv("A_ADAPTER_PATH", "").strip()
        if not adapter_path:
            default_path = Path("outputs/aagent-lora")
            if default_path.exists():
                adapter_path = str(default_path)

        if adapter_path:
            print(f"[A-Agent] Loading LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        else:
            print("[A-Agent] No adapter found. Using base model.")

        self.model.eval()

    # -------------------------
    # INFERENCE
    # -------------------------
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
            # enable_thinking exists for Qwen; safe to keep
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

        max_new = int(kwargs.get("max_new_tokens", 256))
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
            new_tokens = out_ids[len(in_ids):].tolist()
            if tgps_show:
                token_len += len(new_tokens)
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            decoded = _clean_text(decoded)
            outs.append(decoded)

        resp = outs[0] if is_single else outs
        if tgps_show:
            return resp, token_len, gen_time
        return resp, None, None

    # -------------------------
    # SFT LoRA TRAINING
    # -------------------------
    def train_sft_lora(
        self,
        data_path: str,
        output_dir: str,
        epochs: int = 2,
        batch_size: int = 1,
        grad_accum: int = 8,
        lr: float = 2e-4,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        rows = _load_rows(data_path)

        formatted_texts = []
        for ex in rows:
            question = str(ex["question"]).strip()
            choices = _normalize_choices(ex["choices"])
            correct_answer = str(ex["answer"]).strip().upper()

            user_prompt = (
                "INSTRUCTIONS:\n"
                "1) Exactly one option is correct.\n"
                "2) Return ONLY valid JSON.\n"
                "3) answer must be one letter: A/B/C/D.\n"
                "4) reasoning <= 60 words.\n\n"
                f"Question: {question}\n"
                f"Choices: {' '.join(choices)}\n\n"
                'Return JSON: {"answer":"A|B|C|D","reasoning":"..."}'
            )

            assistant_output = json.dumps(
                {"answer": correct_answer, "reasoning": "Answer follows from the constraints in the prompt."},
                ensure_ascii=False,
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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
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
            seed=seed,
        )

        # TRL SFTTrainer API varies: pass dataset_text_field/tokenizer only if accepted
        trainer = _make_sft_trainer(
            model=model,
            train_dataset=ds,
            args=training_args,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
        )

        trainer.train()

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[SFT] Saved LoRA adapter to: {output_dir}")

    # -------------------------
    # PPO RL TRAINING (after SFT)
    # -------------------------
    def train_rl_ppo(
        self,
        data_path: str,
        sft_adapter_path: str,
        output_dir: str,
        steps: int = 500,
        batch_size: int = 1,
        mini_batch_size: int = 1,
        lr: float = 1e-6,
        max_new_tokens: int = 96,
        seed: int = 42,
    ):
        torch.manual_seed(seed)

        rows = _load_rows(data_path)
        if not sft_adapter_path or not Path(sft_adapter_path).exists():
            raise ValueError(f"sft_adapter_path not found: {sft_adapter_path}")

        # Load model with value head (for PPO)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Attach SFT LoRA adapter (RL starts from supervised weights)
        model = PeftModel.from_pretrained(model, sft_adapter_path)
        model.eval()

        # Some TRL versions require an explicit ref_model
        ref_model = None
        try:
            ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            ref_model.eval()
        except Exception:
            ref_model = None

        ppo_config = _make_ppo_config(
            learning_rate=lr,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            seed=seed,            # will be ignored if unsupported
        )

        ppo_trainer = _make_ppo_trainer(
            ppo_config,
            model=model,
            ref_model=ref_model,   # ignored if unsupported by installed TRL
            tokenizer=self.tokenizer,
        )

        def build_prompt(ex: Dict[str, Any]) -> str:
            q = str(ex["question"]).strip()
            choices = _normalize_choices(ex["choices"])
            user_prompt = (
                "INSTRUCTIONS:\n"
                "1) Exactly one option is correct.\n"
                "2) Return ONLY valid JSON.\n"
                "3) answer must be one letter: A/B/C/D.\n"
                "4) reasoning <= 60 words.\n\n"
                f"Question: {q}\n"
                f"Choices: {' '.join(choices)}\n\n"
                'Return JSON: {"answer":"A|B|C|D","reasoning":"..."}'
            )
            chat = [
                {"role": "system", "content": "You are an expert quantitative aptitude solver."},
                {"role": "user", "content": user_prompt},
            ]
            return self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        idx = 0
        # PPOTrainer exposes .model (some versions) else .policy_model
        ppo_model = getattr(ppo_trainer, "model", None) or getattr(ppo_trainer, "policy_model", None)
        device = ppo_model.device if ppo_model is not None else model.device

        for step in range(1, steps + 1):
            ex = rows[idx % len(rows)]
            idx += 1

            gt = str(ex["answer"]).strip().upper()
            prompt_text = build_prompt(ex)

            query_toks = self.tokenizer(
                prompt_text, return_tensors="pt", truncation=True, padding=False
            ).input_ids.to(device)

            # Generate from PPO policy
            with torch.no_grad():
                full_toks = ppo_model.generate(
                    input_ids=query_toks,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            gen_toks = full_toks[:, query_toks.shape[1]:]
            out_text = self.tokenizer.decode(gen_toks[0], skip_special_tokens=True)
            out_text = _clean_text(out_text)

            ok, pred, reasoning = _safe_parse_answer_json(out_text)

            reward = 0.0
            if ok:
                reward += 2.0
                if pred == gt:
                    reward += 4.0
                else:
                    reward -= 1.0
                if len(reasoning.split()) > 60:
                    reward -= 0.5
            else:
                reward -= 2.0

            rewards = [torch.tensor(reward, device=device)]

            # PPO update
            try:
                ppo_trainer.step([query_toks[0]], [gen_toks[0]], rewards)
            except TypeError:
                # some versions expect rewards as python floats
                ppo_trainer.step([query_toks[0]], [gen_toks[0]], [reward])

            if step % 25 == 0:
                print(f"[PPO] step={step}/{steps} reward={reward:.2f} pred={pred} gt={gt} ok={ok}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save RL adapter (robust)
        saved = False
        for candidate in [
            getattr(ppo_trainer, "model", None),
            getattr(ppo_trainer, "policy_model", None),
            getattr(ppo_trainer, "pretrained_model", None),
        ]:
            if candidate is None:
                continue
            try:
                candidate.save_pretrained(output_dir)
                saved = True
                break
            except Exception:
                pass

        if not saved:
            # last resort
            try:
                model.save_pretrained(output_dir)
                saved = True
            except Exception as e:
                raise RuntimeError(f"Could not save PPO model to {output_dir}") from e

        self.tokenizer.save_pretrained(output_dir)
        print(f"[PPO] Saved RL adapter to: {output_dir}")


# -------------------------------
# CLI
# -------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="A-Agent: SFT LoRA + PPO RL training")

    parser.add_argument("--mode", type=str, required=True, choices=["sft", "rl", "both"],
                        help="Training mode: sft | rl | both")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to filtered_questions.json (list of dicts)")

    # SFT args
    parser.add_argument("--sft_output_dir", type=str, default="",
                        help="Output dir for SFT LoRA (used in mode=both). If empty, uses --output_dir.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)

    # RL args
    parser.add_argument("--sft_adapter_path", type=str, default="",
                        help="Path to SFT LoRA adapter (required for mode=rl; optional for mode=both)")
    parser.add_argument("--rl_steps", type=int, default=500)
    parser.add_argument("--rl_lr", type=float, default=1e-6)
    parser.add_argument("--rl_max_new_tokens", type=int, default=96)
    parser.add_argument("--rl_batch_size", type=int, default=1)
    parser.add_argument("--rl_mini_batch_size", type=int, default=1)

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (for sft if mode=sft; for rl if mode=rl; for final rl adapter if mode=both)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    agent = AAgent()

    if args.mode == "sft":
        agent.train_sft_lora(
            data_path=args.data_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            seed=args.seed,
        )

    elif args.mode == "rl":
        if not args.sft_adapter_path:
            raise ValueError("--sft_adapter_path is required for mode=rl")
        agent.train_rl_ppo(
            data_path=args.data_path,
            sft_adapter_path=args.sft_adapter_path,
            output_dir=args.output_dir,
            steps=args.rl_steps,
            batch_size=args.rl_batch_size,
            mini_batch_size=args.rl_mini_batch_size,
            lr=args.rl_lr,
            max_new_tokens=args.rl_max_new_tokens,
            seed=args.seed,
        )

    elif args.mode == "both":
        sft_dir = args.sft_output_dir.strip() if args.sft_output_dir.strip() else args.output_dir + "_sft"
        agent.train_sft_lora(
            data_path=args.data_path,
            output_dir=sft_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            seed=args.seed,
        )

        sft_adapter = args.sft_adapter_path.strip() if args.sft_adapter_path.strip() else sft_dir
        agent.train_rl_ppo(
            data_path=args.data_path,
            sft_adapter_path=sft_adapter,
            output_dir=args.output_dir,
            steps=args.rl_steps,
            batch_size=args.rl_batch_size,
            mini_batch_size=args.rl_mini_batch_size,
            lr=args.rl_lr,
            max_new_tokens=args.rl_max_new_tokens,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
