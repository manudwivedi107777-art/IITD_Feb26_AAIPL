#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

from .question_model import QAgent

import random
import json
import re


class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""

    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)

    # -----------------------------
    # Normalization / Validation
    # -----------------------------
    def _strip_choice_prefix(self, s: str) -> str:
        s = "" if s is None else str(s)
        return re.sub(r"^[A-Da-d]\)\s*", "", s).strip()

    def _has_valid_answer(self, q: Dict[str, Any]) -> bool:
        ans = str(q.get("answer", "")).strip().upper()
        return ans in {"A", "B", "C", "D"}

    def _norm(self, s: str) -> str:
        s = "" if s is None else str(s)
        s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    def _signature(self, q: Dict[str, Any]) -> str:
        """
        Dedupe signature based on topic + question + choices (ignores answer/explanation).
        Choice prefixes like 'A) ' are stripped so duplicates match correctly.
        """
        topic = self._norm(q.get("topic", ""))
        question = self._norm(q.get("question", ""))
        choices = q.get("choices", [])
        if not isinstance(choices, list):
            choices = []
        ch_norm = []
        for c in choices:
            ch_norm.append(self._norm(self._strip_choice_prefix(c)))
        return topic + "||" + question + "||" + "||".join(ch_norm)

    # -----------------------------
    # Robust JSON extraction
    # -----------------------------
    def _extract_first_json_object(self, s: str) -> Optional[Dict[str, Any]]:
        """
        Extract first valid JSON object {...} using brace matching.
        Handles extra tokens after JSON safely.
        """
        if not isinstance(s, str):
            return None

        start = s.find("{")
        if start < 0:
            return None

        depth = 0
        in_str = False
        esc = False

        for i in range(start, len(s)):
            ch = s[i]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            if "answer" not in obj and "expected_answer" in obj:
                                obj["answer"] = obj["expected_answer"]
                            if "answer" in obj and isinstance(obj["answer"], str):
                                obj["answer"] = obj["answer"].strip().upper()
                            return obj
                        return None
                    except Exception:
                        return None

        return None

    def _parse_question_obj(self, q: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a question that may be dict or JSON string; tolerate junk after JSON.
        """
        obj = None
        if isinstance(q, dict):
            obj = q
        elif isinstance(q, str):
            try:
                obj = json.loads(q)
            except Exception:
                obj = self._extract_first_json_object(q)

        if not isinstance(obj, dict):
            return None

        if "answer" not in obj and "expected_answer" in obj:
            obj["answer"] = obj["expected_answer"]

        if "answer" in obj and isinstance(obj["answer"], str):
            obj["answer"] = obj["answer"].strip().upper()

        return obj

    # -----------------------------
    # Prompt helpers
    # -----------------------------
    def build_avoid_block(self, prev: List[Dict[str, Any]], max_items: int = 6) -> str:
        """
        Compact 'do-not-repeat' list, kept short to avoid blowing token budget.
        Uses only question + choices (no explanation) to reduce anchoring.
        """
        if not prev:
            return ""

        items = prev[-max_items:]
        lines = []
        for q in items:
            ques = str(q.get("question", "")).strip()
            choices = q.get("choices", [])
            if not isinstance(choices, list):
                choices = []
            ch = [self._strip_choice_prefix(c) for c in choices[:4]]
            ch_txt = " | ".join(ch)
            if ques:
                lines.append(f"- Q: {ques} || C: {ch_txt}")

        if not lines:
            return ""

        return (
            "\n**AVOID DUPLICATES (MUST):**\n"
            "Do NOT generate any question that matches these in story, structure, constraints, or choices:\n"
            + "\n".join(lines)
            + "\n"
        )

    def build_inc_samples(self, inc_samples: List[Dict[str, str]], topic: str) -> str:
        if not inc_samples:
            return ""
        fmt = (
            "EXAMPLE: {}\n"
            "{{\n"
            '  "topic": "{}",\n'
            '  "question": "{}",\n'
            '  "choices": ["A) {}", "B) {}", "C) {}", "D) {}"],\n'
            '  "answer": "{}",\n'
            '  "explanation": "{}"\n'
            "}}"
        )

        sample_str = ""
        for sample in inc_samples:
            question = sample.get("question", "")
            choices = sample.get("choices", [""] * 4)

            cleaned_choices = []
            for c in choices:
                cleaned_choices.append(self._strip_choice_prefix(c))
            while len(cleaned_choices) < 4:
                cleaned_choices.append("")

            answer = sample.get("answer", sample.get("expected_answer", ""))
            explanation = sample.get("explanation", "")
            sample_str += (
                fmt.format(topic, topic.split("/")[-1], question, *cleaned_choices, answer, explanation)
                + "\n\n"
            )
        return sample_str.strip()

    def build_prompt(
        self,
        topic: str,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: Optional[List[Dict[str, str]]] = None,
        avoid_samples: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, str]:
        """Generate an MCQ based question on given topic with specified difficulty"""

        if wadvsys:
            sys_prompt = (
                "You are an expert-level examiner for Quantitative Aptitude and Analytical Reasoning.\n"
                "Output ONLY one valid JSON object.\n"
                "English ONLY.\n"
                "No newline characters anywhere.\n"
                "All keys must be present: topic, question, choices, answer, explanation.\n"
                "The 'answer' value MUST be exactly one of: A, B, C, D (never empty).\n"
                "No extra text before or after the JSON.\n"
                "CRITICAL: Never generate a question that is identical or structurally similar to any previous question.\n"
                "Do NOT reuse the same pattern, same statement structure, same arrangement template, or same series pattern.\n"
                "Each question must be completely new and logically distinct.\n"
            )
        else:
            sys_prompt = (
                "You are an examiner.\n"
                "Output ONLY one valid JSON object.\n"
                "English only.\n"
                "No newline characters.\n"
                "All keys must be present.\n"
                "Answer must be A, B, C, or D.\n"
                "CRITICAL: Do not generate duplicate or structurally similar questions.\n"
            )

        correct_option = random.choice(["A", "B", "C", "D"])
        inc_samples_ex = self.build_inc_samples(inc_samples, topic) if (wicl and inc_samples) else ""
        avoid_block = self.build_avoid_block(avoid_samples or [], max_items=6)

        tmpl = (
            "Generate an EXTREMELY DIFFICULT puzzle-based MCQ on topic: {0}.\n\n"
            "**HARD CONSTRAINTS (MUST FOLLOW):**\n"
            "1) Output ONLY a single valid JSON object (no markdown, no extra text).\n"
            "2) English ONLY. No newline characters anywhere.\n"
            "3) topic <= 4 words (keep short if possible).\n"
            "4) question should be as short as possible; it may end with '?' OR ':' (both acceptable).\n"
            "5) choices: exactly four strings, formatted as A) ... B) ... C) ... D) ...\n"
            "6) Exactly ONE correct answer; 'answer' MUST be ONLY one of A/B/C/D and MUST NOT be empty. Set it to {2}.\n"
            "6.1) The correct choice text MUST match the answer letter. If mismatch occurs, fix answer letter.\n"
            "7) Keep explanation concise.\n"
            "8) ABSOLUTELY NO DUPLICATES: do not reuse any question pattern, statement set, or arrangement from the examples.\n\n"
            "{5}\n"
            "{10}\n"
            "FINAL CHECK (silent): Verify 'answer' is one of A/B/C/D, choices are 4, and output JSON only.\n"
            "RESPONSE FORMAT: Output JSON exactly like:\n"
            "{{"
            "\"topic\":\"{7}\","
            "\"question\":\"...?\","
            "\"choices\":[\"A) ...\",\"B) ...\",\"C) ...\",\"D) ...\"],"
            "\"answer\":\"{8}\","
            "\"explanation\":\"Briefly justify why {9} is correct.\""
            "}}"
        )

        prompt = tmpl.format(
            topic,
            topic,
            correct_option,
            "",
            correct_option,
            inc_samples_ex,
            topic,
            topic.split("/")[-1],
            correct_option,
            correct_option,
            avoid_block,
        )

        return prompt, sys_prompt

    # -----------------------------
    # Checks + Filtering
    # -----------------------------
    def _basic_checks(self, q2: Dict[str, Any]) -> bool:
        """
        Basic validity checks for a generated question object.
        IMPORTANT CHANGE:
          - We DO NOT require '?' anymore (your data has many ':' prompts like "Find the missing term:")
          - We accept '?' OR ':' OR '.)' ending, as long as question is non-empty.
        """
        if "answer" not in q2 and "expected_answer" in q2:
            q2["answer"] = q2["expected_answer"]

        required_keys = ["topic", "question", "choices", "answer", "explanation"]
        if not all((key in q2) for key in required_keys):
            return False

        if not isinstance(q2["choices"], list) or len(q2["choices"]) != 4:
            return False

        if not self._has_valid_answer(q2):
            return False

        # choices must start with A/B/C/D (case-insensitive), like "A) ..."
        checks = all(
            isinstance(choice, str)
            and len(choice.strip()) > 2
            and choice.strip()[0].upper() in "ABCD"
            for choice in q2["choices"]
        )
        if not checks:
            return False

        qtext = str(q2.get("question", "")).strip()
        if not qtext:
            return False

        # allow '?', ':' or generally any meaningful question text
        # (If you REALLY want strict endings, uncomment below)
        # if ("?" not in qtext) and (":" not in qtext):
        #     return False

        return True

    def filter_questions(self, questions: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        parsed: List[Dict[str, Any]] = []
        for q in questions:
            obj = self._parse_question_obj(q)
            if obj and self._basic_checks(obj):
                parsed.append(obj)

        # DEDUPE KEEP FIRST
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for it in parsed:
            sig = self._signature(it)
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(it)

        return deduped

    def save_questions(self, questions: Any, file_path: Union[str, Path]) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=4, ensure_ascii=False)

    # -----------------------------
    # Topic sampling
    # -----------------------------
    def populate_topics(self, topics: Dict[str, List[str]], num_questions: int) -> List[Tuple[str, str]]:
        """
        - sample WITHOUT replacement first (reduces repeated subtopics)
        - if num_questions > unique subtopics, then fill remainder with replacement
        """
        if not isinstance(topics, dict):
            raise ValueError("Topics must be a dictionary with topic names as keys and lists of subtopics as values.")
        all_subtopics = [(t, st) for t, sublist in topics.items() for st in sublist]
        if not all_subtopics:
            raise ValueError("No subtopics found in the provided topics dictionary.")

        if num_questions <= len(all_subtopics):
            return random.sample(all_subtopics, k=num_questions)

        out = random.sample(all_subtopics, k=len(all_subtopics))
        out += random.choices(all_subtopics, k=num_questions - len(all_subtopics))
        return out

    @staticmethod
    def load_icl_samples(file_path: Union[str, Path]) -> Dict[str, List[Dict[str, str]]]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        if not isinstance(samples, dict):
            raise ValueError("Samples must be inside dictionary.")
        return samples

    # -----------------------------
    # Generation (RETRY UNTIL N VALID)
    # -----------------------------
    def generate_batches(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: Optional[Dict[str, List[Dict[str, str]]]] = None,
        max_attempts_multiplier: int = 10,
        **gen_kwargs,
    ) -> Tuple[List[Dict[str, Any]], List[str], List[int | None], List[float | None]]:
        """
        Returns:
          valid_questions: List[dict] (tries to reach len == num_questions)
          raw_outputs: List[str] all raw model outputs attempted
          tls: tokens generated per batch (if tgps_show True, else None entries)
          gts: time per batch generation (if tgps_show True, else None entries)

        This FIXES your issue where you asked for 20 but got 16/4 due to filtering losses.
        We keep generating until we collect N valid (or hit a hard cap).
        """
        target = int(num_questions)
        max_attempts = max(25, target * int(max_attempts_multiplier))

        # Per-subtopic "avoid duplicates" memory (in-run)
        avoid_by_subtopic: Dict[str, List[Dict[str, Any]]] = {}
        # Global dedupe set
        seen_signatures: set[str] = set()

        valid: List[Dict[str, Any]] = []
        raw_outputs: List[str] = []
        tls: List[int | None] = []
        gts: List[float | None] = []

        attempts = 0
        pbar = tqdm(total=target, desc="VALID QUESTIONS", leave=True)

        # We'll cycle topic plans until target reached
        topic_plan = self.populate_topics(topics, max(target, batch_size))
        idx = 0

        while len(valid) < target and attempts < max_attempts:
            if idx >= len(topic_plan):
                # refresh plan
                topic_plan = self.populate_topics(topics, max(target, batch_size))
                idx = 0

            batch_topics = topic_plan[idx : idx + batch_size]
            idx += batch_size

            prompts: List[str] = []
            sys_prompt: Optional[str] = None

            for t in batch_topics:
                sub = t[1]
                inc = inc_samples.get(sub) if (inc_samples and sub in inc_samples) else None
                avoid_samples = avoid_by_subtopic.get(sub, [])
                p, sp = self.build_prompt(
                    f"{t[0]}/{t[1]}",
                    wadvsys=wadvsys,
                    wicl=wicl,
                    inc_samples=inc,
                    avoid_samples=avoid_samples,
                )
                prompts.append(p)
                sys_prompt = sp

            # Generate batch
            resp, tl, gt = self.agent.generate_response(prompts, sys_prompt, **gen_kwargs)
            if not isinstance(resp, list):
                resp = [resp]

            tls.append(tl)
            gts.append(gt)

            for t, out in zip(batch_topics, resp):
                attempts += 1
                raw_outputs.append(out)

                obj = self._parse_question_obj(out)
                if not obj:
                    continue
                if not self._basic_checks(obj):
                    continue

                sig = self._signature(obj)
                if sig in seen_signatures:
                    continue

                # Accept
                seen_signatures.add(sig)
                valid.append(obj)
                pbar.update(1)

                # update avoid memory by subtopic
                sub = t[1]
                avoid_by_subtopic.setdefault(sub, []).append(obj)

                if len(valid) >= target:
                    break

        pbar.close()
        return valid, raw_outputs, tls, gts


if __name__ == "__main__":
    import argparse
    import yaml

    argparser = argparse.ArgumentParser(description="Generate questions using the QuestioningAgent.")
    argparser.add_argument("--num_questions", type=int, default=10, help="Total number of questions to generate.")
    argparser.add_argument("--output_file", type=str, default="outputs/questions.json", help="Output file name.")
    argparser.add_argument("--batch_size", type=int, default=5, help="Batch size for generating questions.")
    argparser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = argparser.parse_args()

    inc_samples = QuestioningAgent.load_icl_samples("assets/topics_example.json")

    with open("assets/topics.json", "r", encoding="utf-8") as f:
        topics = json.load(f)

    agent = QuestioningAgent()

    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml", "r", encoding="utf-8") as f:
        gen_kwargs.update(yaml.safe_load(f))

    valid_questions, raw_outputs, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=True,
        inc_samples=inc_samples,
        **gen_kwargs,
    )

    print(f"Requested {args.num_questions} | Valid saved: {len(valid_questions)} | Raw attempts: {len(raw_outputs)}")

    if args.verbose:
        for q in valid_questions:
            print(json.dumps(q, ensure_ascii=False), flush=True)

        print("\n" + "=" * 50 + "\n")
        if gen_kwargs.get("tgps_show", False):
            safe_gts = [x for x in gts if isinstance(x, (int, float)) and x > 0]
            safe_tls = [x for x in tls if isinstance(x, int)]
            if safe_gts and safe_tls:
                print(
                    f"Total Time Taken: {sum(safe_gts):.3f} seconds; Total Tokens: {sum(safe_tls)}; "
                    f"TGPS: {sum(safe_tls)/sum(safe_gts):.3f} tokens/sec"
                )
        print("\n" + "+" * 50 + "\n")

    # Save valid questions as questions.json
    agent.save_questions(valid_questions, args.output_file)

    # filtered_questions.json (dedupe + checks, but now checks are not killing ':' questions)
    filtered_file_name = args.output_file.replace("questions.json", "filtered_questions.json")
    filtered = agent.filter_questions(valid_questions)
    agent.save_questions(filtered, filtered_file_name)

    # raw debug dump (super useful)
    raw_file = args.output_file.replace("questions.json", "raw_questions.json")
    agent.save_questions(raw_outputs, raw_file)

    print(f"Saved to {args.output_file}, {filtered_file_name}, and {raw_file}!")
