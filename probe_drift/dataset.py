import numpy as np

from typing import Iterable, Tuple, List, Optional


class Dataset:
    """
    Container for a ProbeDrift benchmark split.
    """

    def __init__(self, x: List[str], y: List[str], batch_size: int,
                 source_ids: Optional[List[str]] = None):
        """
        Parameters:
            x (List[str]): a list of input texts.
            y (List[str]): a list of output (target) texts. Must have the same length as `x`.
            batch_size (int): the size of the texts batch.
            source_ids (List[str], optional): dataset name per example. Use this outside
                ProbeDrift to configure per-dataset settings (e.g. generation length).
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.source_ids = list(source_ids) if source_ids is not None else None

    def __iter__(self) -> Iterable[Tuple[List[str], List[str]]]:
        """
        Returns:
            Iterable of (input_texts, target_texts) batches.
        """
        for i in range(0, len(self.x), self.batch_size):
            yield self.x[i : i + self.batch_size], self.y[i : i + self.batch_size]

    def __len__(self) -> int:
        """
        Returns:
            int: number of batches in the dataset.
        """
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def select(self, indices: List[int]):
        """
        Shrinks the dataset down to only texts with the specified index.

        Parameters:
            indices (List[int]): indices to keep in the dataset.
        """
        self.x = [self.x[i] for i in indices]
        self.y = [self.y[i] for i in indices]
        if self.source_ids is not None:
            self.source_ids = [self.source_ids[i] for i in indices]
        return self

    def concat(self, x, y, source_ids=None):
        self.x = self.x + x
        self.y = self.y + y
        if self.source_ids is not None and source_ids is not None:
            self.source_ids = self.source_ids + list(source_ids)
        else:
            self.source_ids = None
        return self

    def subsample(self, size: int, seed: int):
        """
        Subsamples the dataset to the provided size.

        Parameters:
            size (int): size of the resulting dataset,
            seed (int): seed to perform random subsampling with.
        """
        np.random.seed(seed)
        if len(self.x) < size:
            indices = list(range(len(self.x)))
        else:
            indices = np.random.choice(len(self.x), size, replace=False)
        return self.select(indices)

    @staticmethod
    def _format_raw_rows(examples, dataset_name, cfg, instruct=False, few_shot_meta=None):
        """
        Format raw field dicts into (x, y) lists using prompt templates from cfg.

        cfg must contain 'prompt' and optionally 'description' and 'few_shot_prompt'.
        instruct controls which formatting branch is used for datasets that differ
        between base and instruct modes (CoQA, MMLU, TriviaQA, MedQuad).
        few_shot_meta contains raw few-shot rows (not pre-formatted strings), formatted
        here using the same cfg prompt so that prompt changes take effect consistently.
        """
        prompt          = cfg['prompt']
        description     = cfg.get('description', '')
        few_shot_prompt = cfg.get('few_shot_prompt', None)
        dn = dataset_name.lower()

        if "xsum" in dn or "cnn" in dn or "samsum" in dn:
            x = [prompt.format(text=row["text"]) for row in examples]
            y = [row["y"] for row in examples]

        elif "coqa" in dn:
            # Pass-through for v1-compatible per-turn entries (no story grouping).
            if examples and "x" in examples[0]:
                return [row["x"] for row in examples], [row["y"] for row in examples]

            def build_prior(questions, answers, tmpl, i):
                out = ""
                for q, a in zip(questions[:i], answers[:i]):
                    out += tmpl.format(question=q, answer=a)
                return out

            x, y = [], []
            for row in examples:
                formatted_description = description.format(story=row["story"])
                for j, (question, answer) in enumerate(zip(row["questions"], row["answers"])):
                    if instruct:
                        assert few_shot_prompt is not None, \
                            "few_shot_prompt required for CoQA instruct mode."
                        prior = build_prior(row["questions"], row["answers"], few_shot_prompt, j)
                        if prior:
                            few_shot_section = (
                                "\n\nHere are a few examples of questions and answers:"
                                + prior
                                + "\n\nNow answer the following question in the same format, "
                                  "continuing from the questions and answers above."
                            )
                        else:
                            few_shot_section = "\n\n"
                    else:
                        few_shot_section = build_prior(
                            row["questions"], row["answers"], prompt, j
                        )
                    x.append(
                        formatted_description
                        + few_shot_section
                        + prompt.format(question=question, answer="")
                    )
                    y.append(answer)

        elif "sciq" in dn:
            x, y = [], []
            for row in examples:
                formatted_description = description.format(context=row["context"])
                x.append(formatted_description + prompt.format(question=row["question"]))
                y.append(row["y"])

        elif "mmlu" in dn:
            x, y = [], []
            for row in examples:
                subj = row["subject"]
                formatted_description = description.format(subject=subj.replace("_", " "))
                formatted_few_shot = ""
                if few_shot_meta and subj in few_shot_meta:
                    if instruct:
                        assert few_shot_prompt is not None
                        formatted_few_shot = "Here are a few examples of questions and answers:\n\n"
                        for fs in few_shot_meta[subj]:
                            formatted_few_shot += (
                                few_shot_prompt.format(
                                    choices=fs["choices"],
                                    question=fs["question"].strip(),
                                    answer=fs["y"],
                                ) + "\n\n"
                            )
                        formatted_few_shot += "Now answer the following question in the same format:\n\n"
                    else:
                        for fs in few_shot_meta[subj]:
                            formatted_few_shot += (
                                prompt.format(
                                    choices=fs["choices"],
                                    question=fs["question"].strip(),
                                    answer=fs["y"],
                                ) + "\n"
                            )
                x.append(
                    formatted_description + "\n\n" + formatted_few_shot
                    + prompt.format(
                        choices=row["choices"],
                        question=row["question"].strip(),
                        answer="",
                    )
                )
                y.append(row["y"])

        elif "trivia_qa" in dn:
            formatted_few_shot = ""
            if few_shot_meta:
                if instruct:
                    assert few_shot_prompt is not None
                    formatted_few_shot = "\n\nHere are a few examples of questions and answers:\n\n"
                    for fs in few_shot_meta:
                        formatted_few_shot += (
                            few_shot_prompt.format(
                                question=fs["question"].strip(), answer=fs["y"]
                            ) + "\n\n"
                        )
                    formatted_few_shot += "Now answer the following question in the same format:\n\n"
                else:
                    for fs in few_shot_meta:
                        formatted_few_shot += (
                            prompt.format(question=fs["question"].strip(), answer=fs["y"]) + "\n"
                        )
            x = [
                formatted_few_shot + prompt.format(question=row["question"], answer="")
                for row in examples
            ]
            y = [row["y"] for row in examples]

        elif "truthful_qa" in dn:
            x = [prompt.format(question=row["question"]) for row in examples]
            y = [row["y"] for row in examples]

        elif "medquad" in dn:
            formatted_few_shot = ""
            if few_shot_meta:
                if instruct:
                    assert few_shot_prompt is not None
                    formatted_few_shot = "\n\nHere are a few examples of questions and answers:\n\n"
                    formatted_few_shot += description
                    for fs in few_shot_meta:
                        formatted_few_shot += (
                            prompt.format(question=fs["question"].strip(), answer=fs["y"]) + "\n"
                        )
                    formatted_few_shot += "Now answer the following question in the same format:\n\n"
                else:
                    for fs in few_shot_meta:
                        formatted_few_shot += (
                            prompt.format(question=fs["question"].strip(), answer=fs["y"]) + "\n"
                        )
            x = [
                formatted_few_shot + prompt.format(question=row["question"], answer="")
                for row in examples
            ]
            y = [row["y"] for row in examples]

        elif "pubmed_qa" in dn:
            x, y = [], []
            for row in examples:
                context = ' '.join(row["contexts"])
                x.append(prompt.format(context=context, question=row["question"]))
                y.append(row["y"])

        else:
            x = [row["x"] for row in examples]
            y = [row["y"] for row in examples]

        return x, y

