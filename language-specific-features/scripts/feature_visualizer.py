import ast
import math
import os
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from const import lang_choices_to_qualified_name, prompt_templates
from loader import load_dataset_specific_rows
from transformers import AutoTokenizer


class DatasetTokenActivationsExtractor:
    @staticmethod
    def extract(
        dataset_lang_to_dataset_token_activations: dict[str, pd.DataFrame],
        feature_index: int,
        model: str,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model)

        global_dataset_examples_info = {}
        global_max_token_act_val = -math.inf
        global_min_token_act_val = math.inf
        global_num_dataset_examples = 0

        for (
            dataset_name,
            dataset_info,
        ) in dataset_lang_to_dataset_token_activations.items():
            lang_to_dataset_token_activations = dataset_info[
                "dataset_token_activations"
            ]
            config = dataset_info["config"]

            dataset_examples_info = {}

            for lang, df in lang_to_dataset_token_activations.items():
                # Check if feature_index exists in the dataframe
                df_feature_index = df[df["index"] == feature_index]

                if df_feature_index.empty:
                    continue

                # [(dataset_row_id), (token_id), (act_val)]
                feature_index_info = ast.literal_eval(
                    df_feature_index["dataset_row_id_token_id_act_val"].values[0]
                )
                dataset_row_ids = DatasetTokenActivationsExtractor.extract_dataset_rows(
                    feature_index_info
                )
                dataset = load_dataset_specific_rows(
                    config["dataset"], lang, config["split"], dataset_row_ids
                )

                prompt_template = prompt_templates[config["dataset"]][lang]

                # Extract token_act_val for each dataset row
                tokens_act_val = DatasetTokenActivationsExtractor.extract_token_act_val(
                    feature_index_info
                )
                max_token_act_val = (
                    DatasetTokenActivationsExtractor.get_max_active_token_act_val(
                        feature_index_info
                    )
                )
                min_token_act_val = (
                    DatasetTokenActivationsExtractor.get_min_active_token_act_val(
                        feature_index_info
                    )
                )

                # Update global max and min token activation values
                global_max_token_act_val = max(
                    global_max_token_act_val, max_token_act_val
                )
                global_min_token_act_val = min(
                    global_min_token_act_val, min_token_act_val
                )

                # Process each dataset row
                dataset_example_info = []

                for row_id, row in zip(dataset_row_ids, dataset):
                    prompt = prompt_template.format_map(row)
                    prompt_id = tokenizer(prompt)["input_ids"]

                    tokenized_prompt = tokenizer.convert_ids_to_tokens(prompt_id)
                    tokenized_prompt = [
                        tokenizer.convert_tokens_to_string([token])
                        for token in tokenized_prompt
                    ]

                    dataset_example_info.append(
                        {
                            "dataset_name": dataset_name,
                            "dataset_row_id": row_id,
                            "tokenized_prompt": tokenized_prompt,
                            "tokens_act_val": tokens_act_val[row_id],
                            "max_active": DatasetTokenActivationsExtractor.get_max_active_token_act_val(
                                tokens_act_val[row_id]
                            ),
                        }
                    )

                num_dataset_examples = len(dataset_example_info)
                global_num_dataset_examples += num_dataset_examples

                sorted_dataset_example_info = sorted(
                    dataset_example_info, key=lambda x: x["max_active"], reverse=True
                )

                dataset_examples_info[lang] = {
                    "dataset_examples": sorted_dataset_example_info,
                    "dataset_name": dataset_name,
                    "num_dataset_examples": num_dataset_examples,
                    "max_token_act_val": max_token_act_val,
                    "min_token_act_val": min_token_act_val,
                }

            global_dataset_examples_info[dataset_name] = dataset_examples_info

        num_endpoints = 11
        interval = sorted(
            np.linspace(
                global_min_token_act_val, global_max_token_act_val, num_endpoints
            ).tolist(),
        )

        global_dataset_examples_info.update(
            {
                "info": {
                    "global_num_dataset_examples": global_num_dataset_examples,
                    "global_max_token_act_val": global_max_token_act_val,
                    "global_min_token_act_val": global_min_token_act_val,
                    "interval": interval,
                }
            }
        )

        return global_dataset_examples_info

    @staticmethod
    def extract_dataset_rows(feature_index_info):
        dataset_row_ids = set()

        for dataset_row_id, *_ in feature_index_info:
            dataset_row_ids.add(dataset_row_id)

        dataset_row_ids = sorted(dataset_row_ids)
        return dataset_row_ids

    @staticmethod
    def extract_token_act_val(feature_index_info):
        row_id_to_token_act_val = defaultdict(list)

        for dataset_row_id, token_id, act_val in feature_index_info:
            row_id_to_token_act_val[dataset_row_id].append((token_id, act_val))

        return row_id_to_token_act_val

    @staticmethod
    def get_max_active_token_act_val(feature_index_info):
        max_active_token = max(feature_index_info, key=lambda x: x[-1])
        max_active_token_value = max_active_token[-1]

        return max_active_token_value

    @staticmethod
    def get_min_active_token_act_val(feature_index_info):
        min_active_token = min(feature_index_info, key=lambda x: x[-1])
        min_active_token_value = min_active_token[-1]

        return min_active_token_value

    @staticmethod
    def convert_to_all_dataset_examples(
        dataset_lang_to_dataset_token_activations, normalized_lang
    ):

        dataset_token_activations_info = dataset_lang_to_dataset_token_activations[
            "info"
        ]

        # Remove the info key
        dataset_lang_to_dataset_token_activations = (
            dataset_lang_to_dataset_token_activations.copy()
        )
        dataset_lang_to_dataset_token_activations.pop("info")

        lang_to_dataset_examples = defaultdict(lambda: defaultdict(list))

        for _, dataset_info in dataset_lang_to_dataset_token_activations.items():
            for lang, dataset_examples_info in dataset_info.items():
                examples = dataset_examples_info["dataset_examples"]
                normalized_lang = lang_choices_to_qualified_name[lang]
                lang_to_dataset_examples[normalized_lang]["dataset_examples"].extend(
                    examples
                )
                lang_to_dataset_examples[normalized_lang][
                    "num_dataset_examples"
                ].append(
                    (
                        dataset_examples_info["dataset_name"],
                        dataset_examples_info["num_dataset_examples"],
                    )
                )

        all_dataset_examples = {}

        all_dataset_examples["all_examples"] = lang_to_dataset_examples
        all_dataset_examples["info"] = dataset_token_activations_info

        return all_dataset_examples


class FeatureIndexVisualizer:
    @staticmethod
    def generate_visualization_html(
        all_dataset_examples,
        examples_per_section,
        feature_index,
        feature_info,
        model,
        model_layer,
        sae_model,
        output_dir: Path,
    ):
        result_template: str = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
            <title>Feature Browser</title>
            <script type="module" crossorigin src="/language-specific-features/assets/index.js"></script>
            <link rel="stylesheet" crossorigin href="/language-specific-features/assets/index.css">
        </head>
        <body>
            <div id="app" class="m-4 flex flex-col gap-10">
            <div class="flex justify-between gap-2">
                <div class="flex flex-col gap-2">
                    <h1 class="font-bold text-2xl">
                        Feature {feature_index}
                    </h1>
                    <ul class="text-gray-500">
                        <li>Language: {lang}</li>
                        <li>Model: {model}</li>
                        <li>Layer: {model_layer}</li>
                        <li>SAE Model: {sae_model}</li>
                        <li>Selected Token Probability: {selected_prob}</li>
                        <li>Entropy: {entropy}</li>
                    </ul>
                </div>
                {activation_range}
                </div>
            </div>
            <div class="flex flex-col gap-2">
                <h2 class="text-xl font-semibold" id="Interpretation">
                Interpretation
                </h2>
                <p>{interpretation}</p>
                <table class="w-full border-collapse text-sm">
                <thead>
                    <tr class="bg-gray-100">
                    <th class="border border-gray-300 px-4 py-2 text-left">
                        Score Type
                    </th>
                    <th class="border border-gray-300 px-4 py-2 text-left">
                        Accuracy
                    </th>
                    <th class="border border-gray-300 px-4 py-2 text-left">
                        Precision
                    </th>
                    <th class="border border-gray-300 px-4 py-2 text-left">Recall</th>
                    <th class="border border-gray-300 px-4 py-2 text-left">
                        F1 score
                    </th>
                    <th class="border border-gray-300 px-4 py-2 text-left">TPR</th>
                    <th class="border border-gray-300 px-4 py-2 text-left">TNR</th>
                    <th class="border border-gray-300 px-4 py-2 text-left">FPR</th>
                    <th class="border border-gray-300 px-4 py-2 text-left">FNR</th>
                    </tr>
                </thead>
                <tbody>
                    {scores}
                </tbody>    
                </table>
            </div>
            <div class="flex flex-col gap-4">
                <h2 class="text-xl font-semibold" id="text-example-for-each-language">
                <a
                    href="#text-example-for-each-language"
                    class="hover:text-amber-900"
                >
                    Text Examples for Each Language
                </a>
                </h2>
                {language_sections}
            </div>
            <div class="flex flex-col gap-4">
                <h2 class="text-xl font-semibold" id="text-example-for-each-interval">
                <a
                    href="#text-example-for-each-interval"
                    class="hover:text-amber-900"
                >
                    Text Examples for Each Interval
                </a>
                </h2>
                {interval_sections}
            </div>
            </div>
        </body>
        </html>

        """

        score_template = """
            <tr class="hover:bg-gray-50">
                <td class="border border-gray-300 p-2">{score_type}</td>
                <td class="border border-gray-300 p-2">{accuracy}</td>
                <td class="border border-gray-300 p-2">{precision}</td>
                <td class="border border-gray-300 p-2">{recall}</td>
                <td class="border border-gray-300 p-2">{f1_score}</td>
                <td class="border border-gray-300 p-2">{true_positive_rate}</td>
                <td class="border border-gray-300 p-2">{true_negative_rate}</td>
                <td class="border border-gray-300 p-2">{false_positive_rate}</td>
                <td class="border border-gray-300 p-2">{false_negative_rate}</td>
            </tr>
        """

        scores = []

        for metric in feature_info["metrics"]:
            scores.append(score_template.format_map(metric))

        activation_range = FeatureIndexVisualizer.activation_range(
            all_dataset_examples["info"]["interval"],
            all_dataset_examples["info"]["global_min_token_act_val"],
            all_dataset_examples["info"]["global_max_token_act_val"],
        )

        language_sections = FeatureIndexVisualizer.language_sections(
            all_dataset_examples,
            examples_per_section,
        )

        interval_sections = FeatureIndexVisualizer.interval_sections(
            all_dataset_examples,
            examples_per_section,
        )

        result_html = result_template.format_map(
            {
                "feature_index": feature_index,
                "model_layer": model_layer,
                "model": model,
                "sae_model": sae_model,
                "activation_range": activation_range,
                "language_sections": language_sections,
                "interval_sections": interval_sections,
                "lang": feature_info["lang"],
                "selected_prob": feature_info["selected_prob"],
                "entropy": feature_info["entropy"],
                "interpretation": feature_info["interpretation"],
                "scores": "".join(scores),
            }
        )

        os.makedirs(output_dir, exist_ok=True)

        with open(
            output_dir / f"feature_{feature_index}.html", "w", encoding="utf-8"
        ) as f:
            f.write(result_html)

    @staticmethod
    def activation_range(
        interval: list[float], min_val: float, max_val: float, rounding_digit: int = 3
    ):
        template = """
        <div class="flex flex-col gap-2">
            <span class="text-xl font-semibold">Activation Range</span>
            <div class="flex gap-1 flex-wrap">
                {interval}
        </div>
        """

        interval_template = """
        <span class="p-0.5 rounded-sm" style="background-color: {bg_color};"
            >{start}-{end}</span
        >
        """
        start_end_tuples = zip(interval[:-1], interval[1:])

        spans = [
            interval_template.format_map(
                {
                    "bg_color": FeatureIndexVisualizer.interpolate_color(
                        min_val, max_val, (start + end) / 2
                    ),
                    "start": round(start, rounding_digit),
                    "end": round(end, rounding_digit),
                }
            )
            for start, end in start_end_tuples
        ]

        return template.format_map({"interval": "\n".join(spans)})

    @staticmethod
    def interpolate_color(min_val, max_val, value):
        ratio = min(1, max(0, (value - min_val) / (max_val - min_val)))

        # Define min (lightest) and max (darkest) colors
        color_min = (255, 243, 232)
        color_max = (224, 84, 8)

        # Compute interpolated color
        interpolated_color = tuple(
            round(min_comp + ratio * (max_comp - min_comp))
            for min_comp, max_comp in zip(color_min, color_max)
        )

        return f"rgb{interpolated_color}"

    @staticmethod
    def language_sections(
        all_dataset_examples,
        examples_per_section,
    ):
        language_section_template = """
        <section class="flex flex-col gap-2">
          <div>
            <h3 class="text-base font-semibold">{language}</h3>
            <ul class="text-sm text-gray-500">
              <li>#examples: {num_dataset_examples}</li>
            </ul>
          </div>
          <ul class="text-sm">
            {examples}
          </ul>
        </section>
        """

        example_template = """
        <li class="text-[0px]">
            <span class="text-sm py-0.5 rounded-sm font-semibold"
            >{dataset_row_id}.&nbsp;</span
            >
            {tokens}
        </li>
        <hr class="my-2 text-gray-200" />
        """

        token_bg_template = """
        <span
            class="text-sm py-0.5 rounded-sm"
            style="background-color: {bg_color};"
            >{token}</span
            >
        """

        token_template = """
        <span
            class="text-sm py-0.5 rounded-sm"
            >{token}</span
            >
        """

        language_sections = []

        lang_to_dataset_examples_info = all_dataset_examples["all_examples"]

        processed_langs = lang_to_dataset_examples_info.keys()

        for lang in processed_langs:
            examples = []

            dataset_examples = lang_to_dataset_examples_info[lang]["dataset_examples"]
            top_dataset_examples = dataset_examples[:examples_per_section]

            for example in top_dataset_examples:
                tokens = example["tokenized_prompt"].copy()
                token_act_val_ids = [
                    token_id for token_id, _ in example["tokens_act_val"]
                ]
                tokens = [
                    (
                        token_template.format_map(
                            {
                                "token": FeatureIndexVisualizer.escape_whitespace(
                                    token
                                ),
                            }
                        )
                        if token_id not in token_act_val_ids
                        else FeatureIndexVisualizer.escape_whitespace(token)
                    )
                    for token_id, token in enumerate(tokens)
                ]

                for token_id, act_val in example["tokens_act_val"]:
                    tokens[token_id] = token_bg_template.format_map(
                        {
                            "bg_color": FeatureIndexVisualizer.interpolate_color(
                                all_dataset_examples["info"][
                                    "global_min_token_act_val"
                                ],
                                all_dataset_examples["info"][
                                    "global_max_token_act_val"
                                ],
                                act_val,
                            ),
                            "token": tokens[token_id],
                        }
                    )

                example_html = example_template.format_map(
                    {
                        "dataset_row_id": f"{example['dataset_name']}-{example['dataset_row_id']}",
                        "tokens": "".join(tokens),
                    }
                )

                examples.append(example_html)

            language_section_html = language_section_template.format_map(
                {
                    "language": lang,
                    "examples": "".join(examples),
                    "num_dataset_examples": lang_to_dataset_examples_info[lang][
                        "num_dataset_examples"
                    ],
                }
            )

            language_sections.append(language_section_html)

        language_sections_html = "".join(language_sections)

        return language_sections_html

    @staticmethod
    def interval_sections(
        all_dataset_examples,
        examples_per_section,
        rounding_digit: int = 3,
    ):
        interval_section_template = """
        <section class="flex flex-col gap-2">
          <div>
            <h3 class="text-base font-semibold">{interval}</h3>
            <ul class="text-sm text-gray-500">
              <li>Range: {start}-{end}</li>
              <li>#examples: {num_dataset_examples}</li>
            </ul>
          </div>
          <ul class="text-sm">
            {examples}
          </ul>
        </section>
        """

        example_template = """
        <li class="text-[0px]">
            <span class="text-sm py-0.5 rounded-sm font-semibold"
            >{dataset_row_id}.&nbsp;</span
            >
            {tokens}
        </li>
        <hr class="my-2 text-gray-200" />
        """

        token_bg_template = """
        <span
            class="text-sm py-0.5 rounded-sm"
            style="background-color: {bg_color};"
            >{token}</span
            >
        """

        token_template = """
        <span
            class="text-sm py-0.5 rounded-sm"
            >{token}</span
            >
        """

        lang_to_dataset_examples_info = all_dataset_examples["all_examples"]

        # Put examples into corresponding interval
        interval_to_examples, interval = (
            FeatureIndexVisualizer.extract_interval_examples(
                all_dataset_examples["info"]["interval"],
                lang_to_dataset_examples_info,
                rounding_digit,
            )
        )

        # Generate HTML for each interval
        interval_sections = []

        start_endpoint = interval[:-1]
        start_endpoint.reverse()

        end_endpoint = interval[1:]
        end_endpoint.reverse()

        interval_tuples = zip(start_endpoint, end_endpoint)

        for index, interval in enumerate(interval_tuples, start=1):
            examples = []

            dataset_examples = sorted(
                interval_to_examples[interval],
                key=lambda x: x["max_active"],
                reverse=True,
            )
            top_dataset_examples = dataset_examples[:examples_per_section]

            for example in top_dataset_examples:
                tokens = example["tokenized_prompt"].copy()
                token_act_val_ids = [
                    token_id for token_id, _ in example["tokens_act_val"]
                ]
                tokens = [
                    (
                        token_template.format_map(
                            {
                                "token": FeatureIndexVisualizer.escape_whitespace(
                                    token
                                ),
                            }
                        )
                        if token_id not in token_act_val_ids
                        else FeatureIndexVisualizer.escape_whitespace(token)
                    )
                    for token_id, token in enumerate(tokens)
                ]

                for token_id, act_val in example["tokens_act_val"]:
                    tokens[token_id] = token_bg_template.format_map(
                        {
                            "bg_color": FeatureIndexVisualizer.interpolate_color(
                                all_dataset_examples["info"][
                                    "global_min_token_act_val"
                                ],
                                all_dataset_examples["info"][
                                    "global_max_token_act_val"
                                ],
                                act_val,
                            ),
                            "token": tokens[token_id],
                        }
                    )

                example_html = example_template.format_map(
                    {
                        "dataset_row_id": f"{example['dataset_name']}-{example['dataset_row_id']}",
                        "tokens": "".join(tokens),
                    }
                )

                examples.append(example_html)

            inteval_section_html = interval_section_template.format_map(
                {
                    "interval": f"interval {index}",
                    "start": interval[0],
                    "end": interval[1],
                    "examples": "".join(examples),
                    "num_dataset_examples": len(interval_to_examples[interval]),
                }
            )

            interval_sections.append(inteval_section_html)

        interval_sections_html = "".join(interval_sections)

        return interval_sections_html

    @staticmethod
    def extract_interval_examples(
        interval: list[float], lang_to_dataset_examples_info, rounding_digit
    ):
        interval_to_examples = defaultdict(list)

        interval = list(
            map(
                lambda endpoint: round(endpoint, rounding_digit),
                interval,
            )
        )

        processed_langs = lang_to_dataset_examples_info.keys()

        for lang in processed_langs:
            dataset_examples = lang_to_dataset_examples_info[lang]["dataset_examples"]

            for example in dataset_examples:
                max_active = example["max_active"]
                interval_end_idx = bisect_left(interval, max_active)
                interval_start_idx = interval_end_idx - 1

                interval_end = interval[interval_end_idx]
                interval_start = interval[interval_start_idx]

                interval_to_examples[(interval_start, interval_end)].append(example)

        return interval_to_examples, interval

    @staticmethod
    def escape_whitespace(token):
        return token.replace(" ", "&nbsp;")


def generate_feature_activations_visualization(
    dataset_lang_to_dataset_token_activations: dict[str, pd.DataFrame],
    feature_index: int,
    feature_info: dict,
    model: str,
    model_layer: str,
    sae_model: str,
    output_dir: Path,
    iso_lang_to_qualified_lang_name,
    examples_per_section: int = 10,
):
    dataset_lang_to_dataset_examples_info = DatasetTokenActivationsExtractor.extract(
        dataset_lang_to_dataset_token_activations, feature_index, model
    )

    all_dataset_examples = (
        DatasetTokenActivationsExtractor.convert_to_all_dataset_examples(
            dataset_lang_to_dataset_examples_info, iso_lang_to_qualified_lang_name
        )
    )

    FeatureIndexVisualizer.generate_visualization_html(
        all_dataset_examples,
        examples_per_section,
        feature_index,
        feature_info,
        model,
        model_layer,
        sae_model,
        output_dir,
    )
