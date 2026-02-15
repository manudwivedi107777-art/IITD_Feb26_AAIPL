#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .question_model_RN import QAgent
# from .question_model_llama import QAgent

import random
import json
import re


class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""

    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)

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

    def _extract_first_json_object(self, s: str) -> Dict[str, Any] | None:
        """
        Fast local extraction of first {...} JSON object (avoids calling LLM again).
        Fixes cases like: '{...} DXO' or extra tokens after JSON.
        """
        if not isinstance(s, str):
            return None
        m = re.search(r"\{.*\}", s)
        if not m:
            return None
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                # normalize expected_answer -> answer if present
                if "answer" not in obj and "expected_answer" in obj:
                    obj["answer"] = obj["expected_answer"]
                return obj
            return None
        except Exception:
            return None

    def build_inc_samples(self, inc_samples: List[Dict[str, str]], topic: str) -> str:
        r"""
        Build a string of example questions from the provided samples.
        """
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

            # Prevent "A) A) X" if samples already include prefixes
            cleaned_choices = []
            for c in choices:
                cleaned_choices.append(self._strip_choice_prefix(c))
            while len(cleaned_choices) < 4:
                cleaned_choices.append("")

            answer = sample.get("answer", sample.get("expected_answer", ""))
            explanation = sample.get("explanation", "")
            sample_str += (
                fmt.format(
                    topic, topic.split("/")[-1], question, *cleaned_choices, answer, explanation
                )
                + "\n\n"
            )
        return sample_str.strip()

    def build_prompt(
        self,
        topic: str,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: List[Dict[str, str]] | None = None,
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
                "If uncertain, still choose one option letter and ensure it matches the correct choice.\n"
                "No extra text before or after the JSON.\n"
                "Do NOT repeat any question from the examples or earlier outputs; always create a new one.\n"
            )
        else:
            sys_prompt = (
                "You are an examiner.\n"
                "Output ONLY one valid JSON object.\n"
                "English only.\n"
                "No newline characters.\n"
                "All keys must be present: topic, question, choices, answer, explanation.\n"
                "The 'answer' value MUST be exactly one of: A, B, C, D (never empty).\n"
                "No extra text before or after the JSON.\n"
                "Do NOT repeat any question from the examples or earlier outputs; always create a new one.\n"
            )

        tmpl = (
            "Generate an EXTREMELY DIFFICULT puzzle-based MCQ on topic: {0}.\n\n"
            "**HARD CONSTRAINTS (MUST FOLLOW):**\n"
            "1) Output ONLY a single valid JSON object (no markdown, no extra text).\n"
            "2) English ONLY. No newline characters anywhere.\n"
            "3) topic <= 4 words (keep short if possible).\n"
            "4) question should be as short as possible and ends with '?'.\n"
            "5) choices: exactly four strings, formatted as A) ... B) ... C) ... D) ...\n"
            "6) Exactly ONE correct answer; 'answer' MUST be ONLY one of A/B/C/D and MUST NOT be empty. Set it to {2}.\n"
            "6.1) The correct choice text MUST match the answer letter. If mismatch occurs, fix answer letter.\n"
            "7) Try to keep [topic+question+choices+answer] under 150 tokens if possible, but correctness is priority.\n"
            "8) explanation should be concise.\n"
            "9) ABSOLUTELY NO DUPLICATES: do not reuse any question pattern, statement set, or arrangement from the examples.\n\n"
            "**QUALITY RULES:**\n"
            "- Fully specified (no missing info).\n"
            "- No contradictions.\n"
            "- Plausible distractors.\n\n"
            "{5}"
            "FINAL CHECK (silent): Verify 'answer' is one of A/B/C/D and output JSON only.\n"
            "RESPONSE FORMAT: Strictly output JSON exactly like:\n"
            "{{\n"
            '  "topic": "{7}",\n'
            '  "question": "...?",\n'
            '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
            '  "answer": "{8}",\n'
            '  "explanation": "Briefly justify why {9} is correct."\n'
            "}}"
        )

        correct_option = random.choice(["A", "B", "C", "D"])

        if wicl:
            inc_samples_ex = self.build_inc_samples(inc_samples, topic)
        else:
            inc_samples_ex = ""

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
        )

        return prompt, sys_prompt

    def generate_question(
        self,
        topic: Tuple[str, str] | List[Tuple[str, str]],
        wadvsys: bool,
        wicl: bool,
        inc_samples: Dict[str, List[Dict[str, str]]] | None,
        **gen_kwargs,
    ) -> Tuple[List[str], int | None, float | None]:
        """Generate a question prompt for the LLM"""
        if isinstance(topic, list):
            prompt = []
            for t in topic:
                p, sp = self.build_prompt(
                    f"{t[0]}/{t[1]}", wadvsys, wicl, inc_samples[t[1]]
                )
                prompt.append(p)
        else:
            prompt, sp = self.build_prompt(
                f"{topic[0]}/{topic[1]}", wadvsys, wicl, inc_samples[topic[1]]
            )

        resp, tl, gt = self.agent.generate_response(prompt, sp, **gen_kwargs)

        if (isinstance(resp, list) and all(isinstance(r, str) for r in resp)) or isinstance(resp, str):
            return resp, tl, gt
        else:
            return (
                "",
                tl,
                gt if not isinstance(resp, list) else [""] * len(resp),
                tl,
                gt,
            )

    def generate_batches(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: Dict[str, List[Dict[str, str]]] | None = None,
        **kwargs,
    ) -> Tuple[List[str], List[int | None], List[float | None]]:
        extended_topics = self.populate_topics(topics, num_questions)
        questions = []
        tls, gts = [], []

        total_batches = (len(extended_topics) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ")

        for i in range(0, len(extended_topics), batch_size):
            batch_topics = extended_topics[i : i + batch_size]
            batch_questions = self.generate_question(
                batch_topics, wadvsys, wicl, inc_samples, **kwargs
            )
            questions.extend(batch_questions[0]), tls.append(batch_questions[1]), gts.append(batch_questions[2])
            pbar.update(1)

        if len(extended_topics) % batch_size != 0:
            batch_topics = extended_topics[-(len(extended_topics) % batch_size) :]
            batch_questions = self.generate_question(
                batch_topics, wadvsys, wicl, inc_samples, **kwargs
            )
            questions.extend(batch_questions[0]), tls.append(batch_questions[1]), gts.append(batch_questions[2])
            pbar.update(1)

        pbar.close()
        return questions, tls, gts

    def count_tokens_q(self, text: str) -> int:
        if not hasattr(self.agent, "tokenizer"):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    # ✅ FILTER: NO token filtering, BUT DEDUPE KEEP FIRST
    def filter_questions(self, questions: List[str | Dict[str, Any]]) -> List[Dict[str, Any]]:
        def basic_checks(q2: Dict[str, Any]) -> bool:
            # normalize expected_answer -> answer
            if "answer" not in q2 and "expected_answer" in q2:
                q2["answer"] = q2["expected_answer"]

            required_keys = ["topic", "question", "choices", "answer"]
            if not all((key in q2) for key in required_keys):
                return False

            if not isinstance(q2["choices"], list) or len(q2["choices"]) != 4:
                return False

            if not self._has_valid_answer(q2):
                return False

            checks = all(
                isinstance(choice, str)
                and len(choice) > 2
                and choice[0].upper() in "ABCD"
                for choice in q2["choices"]
            )
            if not checks:
                return False

            qtext = str(q2.get("question", "")).strip()
            if not qtext or "?" not in qtext:
                return False

            return True

        parsed: List[Dict[str, Any]] = []
        for q in questions:
            if isinstance(q, dict):
                if basic_checks(q):
                    parsed.append(q)
            elif isinstance(q, str):
                try:
                    q1 = json.loads(q)
                    if basic_checks(q1):
                        parsed.append(q1)
                except json.JSONDecodeError:
                    q2 = self._extract_first_json_object(q)
                    if q2 is not None and basic_checks(q2):
                        parsed.append(q2)

        # ✅ DEDUPE KEEP FIRST
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for it in parsed:
            sig = self._signature(it)
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(it)

        return deduped

    def save_questions(self, questions: Any, file_path: str | Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=4, ensure_ascii=False)

    def populate_topics(self, topics: Dict[str, List[str]], num_questions: int) -> List[str]:
        if not isinstance(topics, dict):
            raise ValueError("Topics must be a dictionary with topic names as keys and lists of subtopics as values.")
        all_subtopics = [(t, st) for t, sublist in topics.items() for st in sublist]
        if not all_subtopics:
            raise ValueError("No subtopics found in the provided topics dictionary.")
        selected_topics = random.choices(all_subtopics, k=num_questions)
        return selected_topics

    @staticmethod
    def load_icl_samples(file_path: str | Path) -> Dict[str, List[Dict[str, str]]]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        if not isinstance(samples, dict):
            raise ValueError("Samples must be inside dictionary.")
        return samples


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

    question, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=True,
        inc_samples=inc_samples,
        **gen_kwargs,
    )

    print(f"Generated {len(question)} questions!")
    if args.verbose:
        for q in question:
            print(q, flush=True)
        print("\n" + "=" * 50 + "\n\n")
        if gen_kwargs.get("tgps_show", False):
            print("Time taken per batch generation:", gts)
            print("Tokens generated per batch:", tls)
            print(
                f"Total Time Taken: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; "
                f"TGPS: {sum(tls)/sum(gts):.3f} tokens/sec\n\n"
            )
        print("\n" + "+" * 50 + "\n")

    # Build clean questions list (dicts only); enforce answer presence at source
    ques: List[Dict[str, Any]] = []
    for q in question:
        obj = None
        if isinstance(q, dict):
            obj = q
        elif isinstance(q, str):
            try:
                obj = json.loads(q)
            except Exception:
                obj = agent._extract_first_json_object(q)

        if not isinstance(obj, dict):
            continue

        if "answer" not in obj and "expected_answer" in obj:
            obj["answer"] = obj["expected_answer"]

        ans = str(obj.get("answer", "")).strip().upper()
        if ans not in {"A", "B", "C", "D"}:
            continue

        ques.append(obj)

    agent.save_questions(ques, args.output_file)

    filtered_file_name = args.output_file.replace("questions.json", "filtered_questions.json")
    filtered = agent.filter_questions(ques)
    agent.save_questions(filtered, filtered_file_name)

    print(f"Saved to {args.output_file} and {filtered_file_name}!")
