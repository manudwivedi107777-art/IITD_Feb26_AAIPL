#!/usr/bin/python3

import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from .answer_model import AAgent


class AnsweringAgent(object):
    r"""Agent responsible for answering MCQ questions"""

    def __init__(self, select_prompt1: bool = True, **kwargs):
        # This will auto-load adapter via A_ADAPTER_PATH or kwargs
        self.agent = AAgent(**kwargs)
        self.select_prompt1 = select_prompt1

    # --------------------------------------------------
    # Prompt Builder (STRICT JSON)
    # --------------------------------------------------
    def build_prompt(self, question_data: Dict[str, Any]) -> Tuple[str, str]:
    
        sys_prompt = (
            "You are an expert quantitative aptitude solver.\n"
            "Think internally but OUTPUT ONLY ONE valid JSON object.\n"
            "No markdown. No extra text. No explanation outside JSON."
        )
    
        tmpl = (
            "INSTRUCTIONS:\n"
            "1) Exactly one option is correct.\n"
            "2) Return ONLY valid JSON.\n"
            "3) answer must be one letter: A/B/C/D.\n"
            "4) reasoning <= 60 words.\n\n"
            "Question: {q}\n"
            "Choices: {c}\n\n"
            "Return JSON exactly like:\n"
            '{{"answer":"A","reasoning":"..."}}'
        )
    
        prompt = tmpl.format(
            q=str(question_data["question"]).strip(),
            c=self._format_choices(question_data["choices"]),
        )
    
        return prompt, sys_prompt


    # --------------------------------------------------
    # Single / Batch Answering
    # --------------------------------------------------
    def answer_question(
        self, question_data: Dict | List[Dict], **kwargs
    ) -> Tuple[List[str] | str, int | None, float | None]:

        if isinstance(question_data, list):
            prompts = []
            sys_prompt = None
            for qd in question_data:
                p, sys_prompt = self.build_prompt(qd)
                prompts.append(p)
            return self.agent.generate_response(prompts, sys_prompt, **kwargs)

        else:
            prompt, sys_prompt = self.build_prompt(question_data)
            return self.agent.generate_response(prompt, sys_prompt, **kwargs)

    def answer_batches(
        self, questions: List[Dict], batch_size: int = 5, **kwargs
    ) -> Tuple[List[str], List[int | None], List[float | None]]:

        answers = []
        tls, gts = [], []

        total_batches = (len(questions) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="ANSWERING")

        for i in range(0, len(questions), batch_size):
            batch = questions[i: i + batch_size]
            resp, tl, gt = self.answer_question(batch, **kwargs)

            if isinstance(resp, list):
                answers.extend(resp)
            else:
                answers.append(resp)

            tls.append(tl)
            gts.append(gt)
            pbar.update(1)

        pbar.close()
        return answers, tls, gts

    # --------------------------------------------------
    # STRICT JSON Extraction
    # --------------------------------------------------
    def _extract_first_json_object(self, s: str) -> Dict[str, Any] | None:
        if not isinstance(s, str):
            return None
        match = re.search(r"\{.*\}", s)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def filter_answers(self, answers: List[str]) -> List[Dict[str, Any] | None]:
        """
        Strict validation:
        - Must parse JSON
        - answer ∈ {A,B,C,D}
        - reasoning <= 60 words
        """

        out = []

        for raw in answers:
            obj = None

            if isinstance(raw, dict):
                obj = raw
            else:
                try:
                    obj = json.loads(raw)
                except Exception:
                    obj = self._extract_first_json_object(raw)

            if not isinstance(obj, dict):
                out.append(None)
                continue

            ans = str(obj.get("answer", "")).strip().upper()
            reasoning = str(obj.get("reasoning", "")).strip()

            if ans not in {"A", "B", "C", "D"}:
                out.append(None)
                continue

            if len(reasoning.split()) > 70:
                out.append(None)
                continue

            out.append({
                "answer": ans,
                "reasoning": reasoning
            })

        return out

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def save_answers(self, answers: Any, file_path: str | Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=4, ensure_ascii=False)

    def _format_choices(self, choices: List[str]) -> str:
        formatted = []
        for idx, choice in enumerate(choices[:4]):
            letter = "ABCD"[idx]
            c = str(choice).strip()
            c = re.sub(r"^[A-Da-d]\)\s*", "", c)
            formatted.append(f"{letter}) {c}")
        return " ".join(formatted)


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Answer questions using A-Agent")

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="outputs/answers.json")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Load trained model automatically via A_ADAPTER_PATH
    agent = AnsweringAgent()

    with open(args.input_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    gen_kwargs = {"tgps_show": True}

    if Path("agen.yaml").exists():
        with open("agen.yaml", "r", encoding="utf-8") as f:
            gen_kwargs.update(yaml.safe_load(f))

    answers, tls, gts = agent.answer_batches(
        questions=questions,
        batch_size=args.batch_size,
        **gen_kwargs,
    )

    # Save raw outputs
    agent.save_answers(answers, args.output_file)

    # Save filtered outputs
    filtered_path = args.output_file.replace("answers.json", "filtered_answers.json")
    filtered = agent.filter_answers(answers)
    agent.save_answers(filtered, filtered_path)

    print(f"Saved raw answers to: {args.output_file}")
    print(f"Saved filtered answers to: {filtered_path}")

    if args.verbose:
        print("Done.")
