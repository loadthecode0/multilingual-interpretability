import math
import os
import re
import textwrap
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import umap
from const import (
    lang_choices_to_iso639_1,
    lang_choices_to_qualified_name,
    language_colors,
    layer_to_index,
)
from loader import load_sae, load_task_df
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from tqdm.auto import tqdm
from utils import get_project_dir


def get_layer_to_lang_count(
    layer_to_statistics: dict[str, pd.DataFrame], layers: list[str], config: dict
) -> dict[str, pd.DataFrame]:
    layer_to_lang_count = {}

    for layer in layers:
        df_feature_index_lang = layer_to_statistics[layer][["index", "lang"]]
        df_feaure_index_lang_count = (
            df_feature_index_lang.groupby("index").count().reset_index()
        )

        df_lang_feature_count = (
            df_feaure_index_lang_count.groupby("lang").count().reset_index()
        )
        df_lang_feature_count.rename(
            columns={"lang": "lang_count", "index": "feature_count"}, inplace=True
        )

        num_latents = config["sae"]["num_latents"]
        total_feature_count = df_lang_feature_count["feature_count"].sum()
        total_nonactivated_feature = num_latents - total_feature_count

        num_feature_count = len(df_lang_feature_count)
        df_lang_feature_count.loc[num_feature_count] = [0, total_nonactivated_feature]
        df_lang_feature_count.sort_values(by="lang_count", inplace=True)

        layer_to_lang_count[layer] = df_lang_feature_count

    return layer_to_lang_count


def combine_layer_to_lang_count(layer_to_lang_count: dict[str, pd.DataFrame]):
    combined_df = None

    for layer, df_lang_feature_count in layer_to_lang_count.items():
        df_lang_feature_count["layer"] = layer

        combined_df = pd.concat([combined_df, df_lang_feature_count], ignore_index=True)

    return combined_df


def plot_layer(
    layer_to_statistics: dict[str, pd.DataFrame], title: str, output_path: Path
):
    fig = px.scatter(
        layer_to_statistics,
        x="index",
        y="count",
        color="lang",
        hover_data=[
            "index",
            "lang",
            "count",
            "avg",
            "q1",
            "median",
            "q3",
            "min_active",
            "max_active",
            "std",
        ],
        title=title,
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_all_layers(layer_to_statistics: dict[str, pd.DataFrame], config: dict):
    for layer in config["layers"]:
        layer_to_statistics = layer_to_statistics[layer]

        model_name = model_name
        sae_model_name = sae_model_name
        dataset_name = dataset_name

        title = f"SAE Features for {layer}\n({model_name} - {sae_model_name} - {dataset_name})"
        output_path = (
            get_project_dir()
            / "visualization"
            / "plot"
            / model_name
            / sae_model_name
            / dataset_name
            / f"{layer}.html"
        )

        plot_layer(layer_to_statistics, title, output_path)


def plot_lang_feature_overlap(
    df_lang_feature_overlap_layer: pd.DataFrame,
    title: str,
    output_path: Path,
    range_y: tuple[int, int],
):
    fig = px.bar(
        df_lang_feature_overlap_layer,
        x="lang_count",
        y="feature_count",
        color="lang_count",
        title=title,
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
        ),
    )

    fig.update_yaxes(range=range_y)

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_combined_lang_feature_overlap(
    df_lang_feature_overlap_layer: pd.DataFrame, title: str, output_path: Path
):
    combined_df = combine_layer_to_lang_count(df_lang_feature_overlap_layer)

    fig = px.bar(
        combined_df, x="lang_count", y="feature_count", color="layer", title=title
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
        ),
    )

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_all_lang_feature_overlap(
    layer_to_statistics: dict[str, pd.DataFrame], config: dict, range_y: tuple[int, int]
):
    df_lang_feature_count_layers = get_layer_to_lang_count(
        layer_to_statistics, config["layers"], config
    )

    model_name = model_name
    sae_model_name = sae_model_name
    dataset_name = dataset_name

    for layer in config["layers"]:
        df_lang_feature_count_layer = df_lang_feature_count_layers[layer]
        title = title = (
            f"SAE Features Overlap for {layer}\n({model_name} - {sae_model_name} - {dataset_name})"
        )
        output_path = (
            get_project_dir()
            / "visualization"
            / "overlap"
            / model_name
            / sae_model_name
            / dataset_name
            / f"{layer}.html"
        )

        plot_lang_feature_overlap(
            df_lang_feature_count_layer, title, output_path, range_y
        )

    title = title = (
        f"SAE Features Overlap \n({model_name} - {sae_model_name} - {dataset_name})"
    )
    output_path = (
        get_project_dir()
        / "visualization"
        / "overlap"
        / model_name
        / sae_model_name
        / dataset_name
        / f"combined.html"
    )
    plot_combined_lang_feature_overlap(df_lang_feature_count_layers, title, output_path)


def plot_lang_feature_overlap_trend(
    layer_to_statistics: dict[str, pd.DataFrame], config: dict
):
    df_lang_feature_count_layers = get_layer_to_lang_count(
        layer_to_statistics, config["layers"], config
    )

    combined_df = combine_layer_to_lang_count(df_lang_feature_count_layers)

    model_name = config["model"].split("/")[-1]
    sae_model_name = config["sae"]["model"].split("/")[-1]
    dataset_name = config["dataset"].split("/")[-1]

    title = f"SAE Features Overlap\n({model_name} - {dataset_name})"

    fig = px.line(
        combined_df,
        x="layer",
        y="feature_count",
        color="lang_count",
        title=title,
        color_discrete_map=language_colors,
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
        ),
    )

    output_path = (
        get_project_dir()
        / "visualization"
        / "trend"
        / model_name
        / sae_model_name
        / dataset_name
        / "sae_features_overlap_trend.html"
    )

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_heatmap(
    df: pd.DataFrame, title: str, output_path: Path, labels: dict[str, str]
):
    fig = px.imshow(
        df,
        labels=labels,
        x=df.columns,
        y=df.index,
        color_continuous_scale="Blues",
        title=title,
        text_auto=True,
    )

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def get_lang_to_feature_indexex(df: pd.DataFrame):
    return df.groupby("lang")["index"].apply(set)


def get_feature_index_to_langs(df: pd.DataFrame):
    return df.groupby("index")["lang"].apply(set)


def get_lang_specific_feature_index(df: pd.DataFrame, specific_feature_lang_count: int):
    feature_index_to_langs = get_feature_index_to_langs(df)
    specific_feature_index_to_langs = feature_index_to_langs[
        feature_index_to_langs.apply(len) == specific_feature_lang_count
    ]
    langs = df["lang"].unique()

    lang_specific_feature_index = {lang: {} for lang in langs}

    for lang in langs:
        lang_specific_feature_index[lang] = set(
            specific_feature_index_to_langs[
                specific_feature_index_to_langs.apply(
                    lambda feature_langs: lang in feature_langs
                )
            ].index
        )

    return pd.Series(lang_specific_feature_index)


def plot_cross_co_occurrence(
    config1: dict,
    df1: pd.DataFrame,
    config2: dict,
    df2: pd.DataFrame,
    title: str,
    output_path: Path,
    specific_feature_lang_count: int | None = None,
):
    lang_to_indices1 = (
        get_lang_to_feature_indexex(df1)
        if specific_feature_lang_count is None
        else get_lang_specific_feature_index(df1, specific_feature_lang_count)
    )
    lang_to_indices2 = (
        get_lang_to_feature_indexex(df2)
        if specific_feature_lang_count is None
        else get_lang_specific_feature_index(df2, specific_feature_lang_count)
    )

    langs1 = config1["languages"]
    langs2 = config2["languages"]

    df_co_occurrence_matrix = pd.DataFrame(
        index=langs1, columns=langs2, dtype=int
    ).fillna(0)

    for lang1 in langs1:
        for lang2 in langs2:
            df_co_occurrence_matrix.loc[lang1, lang2] = len(
                lang_to_indices1[lang1] & lang_to_indices2[lang2]
            )

    dataset1_name = config1["dataset"]
    dataset2_name = config2["dataset"]

    fig = px.imshow(
        df_co_occurrence_matrix,
        labels={
            "y": f"Language ({dataset1_name})",
            "x": f"Language ({dataset2_name})",
            "color": "Shared Indices",
        },
        x=df_co_occurrence_matrix.columns,
        y=df_co_occurrence_matrix.index,
        color_continuous_scale="Blues",
        title=title,
        text_auto=True,
    )

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)

    return df_co_occurrence_matrix


def plot_all_cross_co_occurrence(
    layer_to_statistics1: dict[str, pd.DataFrame],
    config1: dict,
    layer_to_statistics2: dict[str, pd.DataFrame],
    config2: dict,
    specific_feature_lang_count: int | None = None,
):
    overlap_layers = list(filter(lambda x: x in config2["layers"], config1["layers"]))

    df_combined_co_occurrence_matrix = None

    labels = {
        "y": f"Language ({dataset1_name})",
        "x": f"Language ({dataset2_name})",
        "color": "Shared Indices",
    }

    folder_name = (
        "all"
        if specific_feature_lang_count is None
        else f"count-{specific_feature_lang_count}"
    )
    lang_count_info = (
        ""
        if specific_feature_lang_count is None
        else f"Lang Count {specific_feature_lang_count} "
    )

    for layer in overlap_layers:
        layer_to_statistics1 = layer_to_statistics1[layer]
        layer_to_statistics2 = layer_to_statistics2[layer]

        model1_name = model1_name
        sae_model1_name = sae_model1_name
        dataset1_name = dataset1_name

        model2_name = model2_name
        sae_model2_name = sae_model2_name
        dataset2_name = dataset2_name

        title = f"SAE Features Cross Co-Occurrence {lang_count_info}for {layer} ({model1_name} - {sae_model1_name} - {dataset1_name} and {dataset2_name})"

        output_path = (
            get_project_dir()
            / "visualization"
            / "cross-co-occurrence"
            / model1_name
            / sae_model1_name
            / dataset1_name
            / dataset2_name
            / folder_name
            / f"{layer}.html"
        )

        df_co_occurrence_matrix = plot_cross_co_occurrence(
            config1,
            layer_to_statistics1,
            config2,
            layer_to_statistics2,
            title,
            output_path,
            specific_feature_lang_count,
        )

        df_combined_co_occurrence_matrix = (
            df_combined_co_occurrence_matrix + df_co_occurrence_matrix
            if df_combined_co_occurrence_matrix is not None
            else df_co_occurrence_matrix
        )

    title = f"SAE Features Cross Co-Occurrence {lang_count_info} ({model1_name} - {sae_model1_name} - {dataset1_name} and {dataset2_name})"
    output_path = (
        get_project_dir()
        / "visualization"
        / "cross-co-occurrence"
        / model1_name
        / sae_model1_name
        / dataset1_name
        / dataset2_name
        / folder_name
        / f"combined.html"
    )
    plot_heatmap(df_combined_co_occurrence_matrix, title, output_path, labels=labels)


def plot_all_co_occurrence(
    layer_to_statistics: dict[str, pd.DataFrame],
    config: dict,
    specific_feature_lang_count: int | None = None,
):

    df_combined_co_occurrence_matrix = None

    folder_name = (
        "all"
        if specific_feature_lang_count is None
        else f"count-{specific_feature_lang_count}"
    )
    lang_count_info = (
        ""
        if specific_feature_lang_count is None
        else f"Lang Count {specific_feature_lang_count} "
    )

    model_name = config["model"].split("/")[-1]
    sae_model_name = config["sae"]["model"].split("/")[-1]
    dataset_name = config["dataset"].split("/")[-1]

    for layer in config["layers"]:
        layer_to_statistics = layer_to_statistics[layer]
        title = title = (
            f"SAE Features Co-Occurrence {lang_count_info}for {layer}\n({model_name} _ {sae_model_name} - {dataset_name})"
        )
        output_path = (
            get_project_dir()
            / "visualization"
            / "co-occurrence"
            / model_name
            / sae_model_name
            / dataset_name
            / folder_name
            / f"{layer}.html"
        )

        df_co_occurrence_matrix = plot_cross_co_occurrence(
            config,
            layer_to_statistics,
            config,
            layer_to_statistics,
            title,
            output_path,
            specific_feature_lang_count,
        )

        df_combined_co_occurrence_matrix = (
            df_combined_co_occurrence_matrix + df_co_occurrence_matrix
            if df_combined_co_occurrence_matrix is not None
            else df_co_occurrence_matrix
        )

    title = title = (
        f"SAE Features Co-Occurrence\n({model_name} - {sae_model_name} - {dataset_name})"
    )
    output_path = (
        get_project_dir()
        / "visualization"
        / "co-occurrence"
        / model_name
        / sae_model_name
        / dataset_name
        / folder_name
        / f"combined.html"
    )
    plot_heatmap(
        df_combined_co_occurrence_matrix,
        title,
        output_path,
        labels=dict(x=f"Language", y=f"Language", color="Shared Indices"),
    )


def plot_box_plot(
    layer_to_statistics: dict[str, pd.DataFrame],
    layers: list[str],
    lang: str,
    x: str,
    y: str,
    title: str,
    output_path: Path,
):
    fig = go.Figure()

    for i, layer in enumerate(layers):
        layer_to_statistics_lang = layer_to_statistics[layer][
            layer_to_statistics[layer]["lang"] == lang
        ]

        data = layer_to_statistics_lang[y]

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        upper_bound = q3 + 1.5 * iqr

        # Find outliers
        outliers = data[data > upper_bound]

        fig.add_trace(
            go.Box(
                x=layer_to_statistics_lang[x],
                y=data,
                name=layer,
                boxpoints=False,
            )
        )

        ranges = [
            (upper_bound, 1000, "mild"),
            (1000, 10000, "moderate"),
            (10000, float("inf"), "extreme"),
        ]

        number_of_count = len(layer_to_statistics_lang["count"])
        annotation_text = [
            f"{number_of_count} total active features",
        ]
        for lower, upper, label in ranges:
            range_outliers = outliers[(outliers > lower) & (outliers <= upper)]
            if len(range_outliers) > 0:
                annotation_text.append(
                    f"{len(range_outliers)} {label} outliers ({lower:.0f}-{upper if upper != float('inf') else 'inf'}) max={range_outliers.max():.0f}"
                )

        if annotation_text:
            fig.add_annotation(
                x=i,
                y=10,
                text="<br>".join(annotation_text),
                showarrow=False,
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                opacity=0.8,
            )

    fig.update_layout(
        title=title,
        yaxis_type="log",
    )

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_all_count_box_plots(
    layer_to_statistics: dict[str, pd.DataFrame],
    config: dict,
):
    model_name = config["model"].split("/")[-1]
    sae_model_name = config["sae"]["model"].split("/")[-1]
    dataset_name = config["dataset"].split("/")[-1]

    for lang in config["languages"]:
        title = title = (
            f"SAE Features Count Box Plot for {lang}\n({model_name} - {sae_model_name} - {dataset_name})"
        )
        output_path = (
            get_project_dir()
            / "visualization"
            / "count-box-plot"
            / model_name
            / sae_model_name
            / dataset_name
            / f"{lang}.html"
        )

        plot_box_plot(
            layer_to_statistics,
            config["layers"],
            lang,
            "layer",
            "count",
            title,
            output_path,
        )


def plot_lape_result(lape_result: dict, title, out_dir: Path):
    sorted_lang = lape_result["sorted_lang"]

    df_all_langs = None

    for lang_index, lang in enumerate(sorted_lang):
        final_indice = lape_result["final_indice"][lang_index]
        frequency_count = list(
            map(lambda layer_index: layer_index.shape[0], final_indice)
        )
        indices = list(range(len(frequency_count)))

        df_lang = pd.DataFrame(
            {
                "Layer": indices,
                "Count": frequency_count,
                "Lang": lang_choices_to_iso639_1[lang],
            }
        )

        plot_specific_lape_lang(lang, df_lang, title, out_dir)

        df_all_langs = pd.concat([df_all_langs, df_lang], ignore_index=True)

    plot_combined_lape_lang(df_all_langs, title, out_dir)
    plot_combined_lape_lang_count(df_all_langs, title, out_dir)


def plot_specific_lape_lang(lang, df_lang, title, out_dir):
    fig = px.bar(
        df_lang,
        x="Layer",
        y="Count",
        title=f"{title} ({lang})",
        color="Lang",
        labels={"Layer": "Layer", "Count": "Count"},
        color_discrete_map=language_colors,
        text="Count",  # Add this to show the values
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
        )
    )

    fig.update_layout(plot_bgcolor="white")

    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )

    fig.update_layout(showlegend=True)

    output_dir = get_project_dir() / out_dir
    output_path = output_dir / f"lape_{lang}.html"

    os.makedirs(output_dir, exist_ok=True)

    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_combined_lape_lang(df_all_langs, title, out_dir):
    fig = px.bar(
        df_all_langs,
        x="Layer",
        y="Count",
        color="Lang",
        title=title,
        labels={"Layer": "Layer", "Count": "Count"},
        color_discrete_map=language_colors,
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
        )
    )

    fig.update_layout(plot_bgcolor="white")

    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )

    output_dir = get_project_dir() / out_dir
    output_path = output_dir / f"lape.html"

    os.makedirs(output_dir, exist_ok=True)

    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_combined_lape_lang_count(df_all_langs, title, out_dir):
    df_copy = df_all_langs.copy()

    df_copy["Layer"] = df_copy["Layer"].astype(str)

    # Create a color map for layers 0-15
    layer_colors = {
        "0": "#1f77b4",  # blue
        "1": "#ff7f0e",  # orange
        "2": "#2ca02c",  # green
        "3": "#d62728",  # red
        "4": "#9467bd",  # purple
        "5": "#8c564b",  # brown
        "6": "#e377c2",  # pink
        "7": "#7f7f7f",  # gray
        "8": "#bcbd22",  # olive
        "9": "#17becf",  # teal
        "10": "#aec7e8",  # light blue
        "11": "#ffbb78",  # light orange
        "12": "#98df8a",  # light green
        "13": "#ff9896",  # light red
        "14": "#c5b0d5",  # light purple
        "15": "#c49c94",  # light brown
    }

    # Group by Lang and sum Counts to determine sorting order
    lang_totals = df_copy.groupby("Lang")["Count"].sum().reset_index()
    lang_totals = lang_totals.sort_values("Count", ascending=True)
    sorted_langs = lang_totals["Lang"].tolist()

    # Create a categorical column with custom order
    df_copy["Lang"] = pd.Categorical(
        df_copy["Lang"], categories=sorted_langs, ordered=True
    )

    # Sort the dataframe
    df_copy = df_copy.sort_values("Lang")

    # Create figure with sorted x-axis
    # Calculate sum of values for each Lang
    df_sums = df_copy.groupby("Lang")["Count"].sum().reset_index()

    fig = px.bar(
        df_copy,
        x="Lang",
        y="Count",
        color="Layer",
        title=title,
        color_discrete_map=layer_colors,
        category_orders={
            "Layer": [str(i) for i in range(16)]
        },  # Ensure layers are ordered from 0-15
    )

    # Add text annotations for the total counts
    for i, row in df_sums.iterrows():
        fig.add_annotation(
            x=row["Lang"],
            y=row["Count"],
            text=str(int(row["Count"])),
            showarrow=False,
            yshift=10,
            font=dict(size=10, color="black"),
        )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
        )
    )

    fig.update_layout(plot_bgcolor="white")

    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )

    output_dir = get_project_dir() / out_dir
    output_path = output_dir / f"feature_counts.html"

    os.makedirs(output_dir, exist_ok=True)

    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_umap(
    lape_result,
    layers: list[str],
    model_name: str,
    sae_model_name: str,
    output_dir: Path,
    interpretations: dict,
    seed: int = 42,
    working_dir: Path | None = None,
):
    for layer in layers:
        layer_index = layer_to_index[layer]
        sae = load_sae(model_name, sae_model_name, layer)

        all_features = [*range(sae.W_dec.size(0))]
        layer_features = []
        layer_feature_indices = []
        layer_feature_langs = []
        layer_feature_interpretations = []
        layer_feature_entropies = []
        layer_feature_selected_probs = []

        for lang_final_indices, lang_sae_features, lang in zip(
            lape_result["final_indice"],
            lape_result["sae_features"],
            lape_result["sorted_lang"],
        ):
            layer_lang_sae_features = lang_sae_features[layer_index].tolist()
            layer_lang_feature_indices = lang_final_indices[layer_index].tolist()
            layer_lang_feature_langs = [lang_choices_to_iso639_1[lang]] * len(
                layer_lang_sae_features
            )
            layer_lang_feature_interpretations = [
                "<br>".join(
                    textwrap.wrap(interpretations[layer][feature_index], width=40)
                )
                for feature_index in layer_lang_feature_indices
            ]

            selected_probs = lape_result["features_info"][lang]["selected_probs"]
            entropies = lape_result["features_info"][lang]["entropies"]

            layer_lang_feature_entropies = []
            layer_lang_feature_selected_probs = []

            for feature_index in layer_lang_feature_indices:
                arg_index = lape_result["features_info"][lang]["indicies"].index(
                    (layer_index, feature_index)
                )
                entropy = round(entropies[arg_index].item(), ndigits=3)
                selected_prob = round(selected_probs[arg_index].item(), ndigits=3)

                layer_lang_feature_entropies.append(entropy)
                layer_lang_feature_selected_probs.append(selected_prob)

            layer_features.extend(layer_lang_sae_features)
            layer_feature_indices.extend(layer_lang_feature_indices)
            layer_feature_langs.extend(layer_lang_feature_langs)
            layer_feature_interpretations.extend(layer_lang_feature_interpretations)
            layer_feature_entropies.extend(layer_lang_feature_entropies)
            layer_feature_selected_probs.extend(layer_lang_feature_selected_probs)

        # https://github.com/lmcinnes/umap/issues/201
        n_components = 2

        if len(layer_features) <= n_components + 1:
            continue

        reducer = umap.UMAP(
            n_neighbors=15,
            metric="cosine",
            min_dist=0.05,
            n_components=n_components,
            random_state=seed,
        )

        remainding_feature_indices = sorted(
            set(all_features) - set(layer_feature_indices)
        )
        remainding_features = sae.W_dec[remainding_feature_indices].detach().tolist()

        layer_features.extend(remainding_features)
        layer_feature_indices.extend(remainding_feature_indices)
        layer_feature_langs.extend((["other"] * len(remainding_features)))
        layer_feature_interpretations.extend((["-"] * len(remainding_features)))
        layer_feature_entropies.extend((["-"] * len(remainding_features)))
        layer_feature_selected_probs.extend((["-"] * len(remainding_features)))

        embedding = reducer.fit_transform(layer_features)

        # Create a custom color palette that assigns grey to "other" and uses plotly colors for languages
        unique_langs = set(layer_feature_langs)
        color_discrete_map = {}

        # Set "other" to grey
        if "other" in unique_langs:
            color_discrete_map["other"] = "rgba(211, 211, 211, 0.5)"

        fig = px.scatter(
            embedding,
            x=0,
            y=1,
            color=layer_feature_langs,
            title=f"UMAP projection of {model_name} - {sae_model_name} - {layer}",
            hover_data={
                "Feature Index": layer_feature_indices,
                "Entropy": layer_feature_entropies,
                "Selected Prob": layer_feature_selected_probs,
                "Interpretation": layer_feature_interpretations,
            },
            color_discrete_map={**language_colors, **color_discrete_map},
        )

        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")

        output_path = output_dir / layer / "umap.html"

        os.makedirs(output_path.parent, exist_ok=True)

        fig.write_html(
            output_path,
            include_plotlyjs="cdn",
        )

        fig = px.scatter(
            embedding,
            x=0,
            y=1,
            color=layer_feature_langs,
            title=f"UMAP projection of {model_name} - {sae_model_name} - {layer}",
            hover_data={
                "Feature Index": layer_feature_indices,
                "Entropy": layer_feature_entropies,
                "Selected Prob": layer_feature_selected_probs,
                "Interpretation": layer_feature_interpretations,
            },
            color_discrete_map={
                **language_colors,
                **{"other": "rgba(211, 211, 211, 0.02)"},
            },
        )

        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")

        save_image(output_path, fig, working_dir)


def plot_ppl_change_matrix(
    langs,
    normal_ppl_result,
    intervened_ppl_results,
    output_path: Path,
    title: str = "PPL Change Matrix",
    num_examples: int | None = None,
):
    num_langs = len(langs)
    ppl_matrix = torch.zeros(num_langs, num_langs)

    for intervened_lang_index, intervened_lang in enumerate(langs):
        normalized_intervened_lang = lang_choices_to_qualified_name[intervened_lang]
        intervened_ppl_result = intervened_ppl_results[normalized_intervened_lang]

        for lang_index, lang in enumerate(langs):
            normalized_lang = lang_choices_to_qualified_name[lang]
            intervened_ppls = np.array(
                intervened_ppl_result[normalized_lang]["perplexities"][:num_examples]
            )
            normal_ppls = np.array(
                normal_ppl_result[normalized_lang]["perplexities"][:num_examples]
            )
            avg_diff = np.mean(intervened_ppls - normal_ppls).round(decimals=2).item()
            ppl_matrix[intervened_lang_index][lang_index] = avg_diff

    # Create custom text matrix with formatted values
    text_matrix = []

    for i in range(num_langs):
        row = []

        for j in range(num_langs):
            value = ppl_matrix[i][j].item()

            if value >= 1000:
                exponent = int(math.log10(value))
                sign = "-" if value < 0 else ""
                formatted_text = f"{sign}10<sup>{exponent}</sup>"
                pass
            else:
                formatted_text = f"{value:.1f}"

            row.append(formatted_text)

        text_matrix.append(row)

    iso_langs = [lang_choices_to_iso639_1[lang] for lang in langs]

    fig = px.imshow(
        ppl_matrix.numpy(),
        x=iso_langs,
        y=iso_langs,
        labels=dict(x="Impacted Language", y="Intervened Language", color="Value"),
        color_continuous_scale=["white", "orange"],
        zmin=0,
        zmax=1000,
        aspect="auto",
    )

    # Add custom text annotations
    for i in range(num_langs):
        for j in range(num_langs):
            fig.add_annotation(
                x=j,
                y=i,
                text=text_matrix[i][j],
                showarrow=False,
                font=dict(size=10, color="black"),
            )

    # Add border to diagonal cells
    for i in range(len(iso_langs)):
        fig.add_shape(
            type="rect",
            x0=i - 0.5,
            y0=i - 0.5,
            x1=i + 0.5,
            y1=i + 0.5,
            line=dict(color="lightgray", width=1),
        )

    fig.update_layout(xaxis_side="top")

    os.makedirs(output_path.parent, exist_ok=True)

    fig.write_html(output_path, include_plotlyjs="cdn")

    fig.update_xaxes(tickfont=dict(size=16), title_text=None)
    fig.update_yaxes(tickfont=dict(size=16), title_text=None)
    fig.update_traces(textfont_size=6)
    fig.update_coloraxes(showscale=False)

    save_image(output_path, fig)


def generate_ppl_change_matrix(
    configs: list[str],
    model: str,
    dataset: str,
    langs: list[str],
    input_dir: Path,
    normal_ppl_result,
):
    iso_lang = [lang_choices_to_iso639_1[lang] for lang in langs]

    for config in configs:
        in_path = input_dir / config

        intervened_sae_features_ppl_results = {
            lang_choices_to_qualified_name[intervened_lang]: torch.load(
                in_path / f"ppl_{intervened_lang}.pt", weights_only=False
            )
            for intervened_lang in langs
        }

        out_path = (
            get_project_dir()
            / "visualization"
            / "ppl"
            / model
            / dataset
            / "sae_intervention"
            / config
            / "ppl_change_matrix.html"
        )

        example_lang = lang_choices_to_qualified_name[langs[0]]
        num_examples = len(
            intervened_sae_features_ppl_results[example_lang][example_lang][
                "perplexities"
            ]
        )

        plot_ppl_change_matrix(
            iso_lang,
            normal_ppl_result,
            intervened_sae_features_ppl_results,
            out_path,
            title=f"PPL Change Matrix for SAE Features Interventions ({num_examples} examples)",
            num_examples=num_examples,
        )


def save_image(
    path: Path, fig, working_dir: Path | None = None, title_keep: bool = False
):
    project_dir = get_project_dir() if working_dir is None else working_dir

    relative_path = path.relative_to(project_dir)

    output_path = project_dir / "images" / relative_path
    output_path = output_path.with_suffix(".pdf")

    os.makedirs(output_path.parent, exist_ok=True)

    if not title_keep:
        fig.update_layout(title=None)

    fig.write_image(output_path)


def plot_metrics(metric, output_dir: Path):
    classes = metric["classes"]
    precision = metric["precision"]
    recall = metric["recall"]
    f1 = metric["f1"]
    confusion_matrix = np.array(metric["confusion_matrix"])

    # Bar plot for per-class metrics
    fig = go.Figure()
    fig.add_trace(go.Bar(x=classes, y=precision, name="Precision"))
    fig.add_trace(go.Bar(x=classes, y=recall, name="Recall"))
    fig.add_trace(go.Bar(x=classes, y=f1, name="F1"))

    fig.update_layout(
        barmode="group",
        title="Per-Class Precision, Recall, and F1 Score",
        yaxis_title="Score",
        xaxis_title="Class",
        yaxis=dict(range=[0, 1.05]),
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = output_dir / "metrics.html"

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
    )

    save_image(output_path, fig)

    # Confusion matrix heatmap
    fig_cm = px.imshow(
        confusion_matrix,
        x=classes,
        y=classes,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="Count"),
        title="Confusion Matrix",
        text_auto=True,
        aspect="auto",
    )
    fig_cm.update_xaxes(side="top")

    output_path = output_dir / "confusion_matrix.html"

    fig_cm.write_html(
        output_path,
        include_plotlyjs="cdn",
    )

    save_image(output_path, fig_cm)


# Raw data from the table
class SimilarityHeatmapGenerator:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
        self.unique_layers = self._get_unique_layers()
        self.layer_boundaries = None

    def _get_layer_number(self, layer_name):
        """Extract layer number from layer name"""
        match = re.search(r"layers\.(\d+)\.", layer_name)
        return int(match.group(1)) if match else 0

    def _get_unique_layers(self):
        """Get unique layers sorted by layer number"""
        layers = list(set(list(self.df["layer1"]) + list(self.df["layer2"])))
        return sorted(layers, key=self._get_layer_number)

    def _get_indices_for_layer(self, layer):
        """Get all indices for a specific layer"""
        indices = set()
        mask1 = self.df["layer1"] == layer
        mask2 = self.df["layer2"] == layer

        indices.update(self.df[mask1]["index1"].tolist())
        indices.update(self.df[mask2]["index2"].tolist())

        return sorted(list(indices))

    def _create_matrix_data(self, metric="iou"):
        """Create matrix data organized by layers"""
        # Create layer-index combinations
        layer_indices = {}
        for layer in self.unique_layers:
            layer_indices[layer] = self._get_indices_for_layer(layer)

        # Create flat list of all combinations, sorted by layer then index
        all_combinations = []
        for layer in self.unique_layers:
            for index in layer_indices[layer]:
                all_combinations.append({"layer": layer, "index": index})

        size = len(all_combinations)
        matrix = np.full((size, size), np.nan)

        # Create labels for axes
        labels = []
        for combo in all_combinations:
            layer_num = self._get_layer_number(combo["layer"])
            labels.append(f"L{layer_num} ({combo['index']})")

        # Fill diagonal with 1.0 (perfect similarity with self)
        np.fill_diagonal(matrix, 1.0)

        # Fill matrix with data
        for _, row in self.df.iterrows():
            # Find positions in matrix
            source_pos = None
            target_pos = None

            for i, combo in enumerate(all_combinations):
                if combo["layer"] == row["layer1"] and combo["index"] == row["index1"]:
                    source_pos = i
                if combo["layer"] == row["layer2"] and combo["index"] == row["index2"]:
                    target_pos = i

            if source_pos is not None and target_pos is not None:
                matrix[source_pos, target_pos] = row[metric]
                matrix[target_pos, source_pos] = row[metric]  # Make symmetric

        return matrix, labels, all_combinations, layer_indices

    def _get_layer_boundaries(self, layer_indices):
        """Calculate layer boundaries for box highlights"""
        boundaries = []
        current_index = 0

        for layer in self.unique_layers:
            layer_size = len(layer_indices[layer])
            boundaries.append(
                {
                    "layer": layer,
                    "layer_num": self._get_layer_number(layer),
                    "start": current_index,
                    "end": current_index + layer_size - 1,
                    "size": layer_size,
                }
            )
            current_index += layer_size

        return boundaries

    def _create_layer_shapes(self, matrix_size):
        """Create shapes for layer boundary boxes"""
        shapes = []

        for boundary in self.layer_boundaries:
            # Rectangle outline for each layer block
            shapes.extend(
                [
                    # Top horizontal line
                    dict(
                        type="line",
                        x0=-0.5,
                        x1=matrix_size - 0.5,
                        y0=boundary["start"] - 0.5,
                        y1=boundary["start"] - 0.5,
                        line=dict(color="white", width=3),
                    ),
                    # Bottom horizontal line
                    dict(
                        type="line",
                        x0=-0.5,
                        x1=matrix_size - 0.5,
                        y0=boundary["end"] + 0.5,
                        y1=boundary["end"] + 0.5,
                        line=dict(color="white", width=3),
                    ),
                    # Left vertical line
                    dict(
                        type="line",
                        x0=boundary["start"] - 0.5,
                        x1=boundary["start"] - 0.5,
                        y0=-0.5,
                        y1=matrix_size - 0.5,
                        line=dict(color="white", width=3),
                    ),
                    # Right vertical line
                    dict(
                        type="line",
                        x0=boundary["end"] + 0.5,
                        x1=boundary["end"] + 0.5,
                        y0=-0.5,
                        y1=matrix_size - 0.5,
                        line=dict(color="white", width=3),
                    ),
                ]
            )

        return shapes

    def _create_heatmap_figure(
        self,
        matrix,
        labels,
        layer_indices,
        metric,
        title=None,
        colorscale=None,
        zmax=None,
        zmin=None,
    ):
        """Create the base heatmap figure with appropriate styling"""
        # Determine what values to show in the hovertemplate based on the metric
        hover_metric_text = "IoU" if metric == "iou" else "Pearson r"

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                showscale=True,
                hoverongaps=False,
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>"
                + hover_metric_text
                + ": %{z:.4f}<extra></extra>",
                colorscale=colorscale,
                zmax=zmax,
                zmin=zmin,
            )
        )

        # Calculate layer boundaries
        self.layer_boundaries = self._get_layer_boundaries(layer_indices)

        # Add layer boundary shapes
        shapes = self._create_layer_shapes(len(matrix))

        # Set default title if none provided
        default_title = f"Layer Similarity Heatmap - {hover_metric_text}"

        # Update layout
        fig.update_layout(
            title={
                "text": title or default_title,
                "x": 0.5,
                "font": {"size": 16},
            },
            xaxis=dict(
                title="Target Layer (Index)",
                tickangle=90,
                side="bottom",
                tickmode="array",
                tickvals=list(range(len(labels))),
                ticktext=labels,
                showticklabels=True,
            ),
            yaxis=dict(
                title="Source Layer (Index)",
                autorange="reversed",
                tickmode="array",
                tickvals=list(range(len(labels))),
                ticktext=labels,
                showticklabels=True,
            ),
            shapes=shapes,
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=800,
            height=800,
        )

        if matrix.shape[0] > 80:
            fig.update_xaxes(tickfont=dict(size=6))
            fig.update_yaxes(tickfont=dict(size=6))
        else:
            fig.update_xaxes(tickfont=dict(size=8))
            fig.update_yaxes(tickfont=dict(size=8))

        return fig

    def create_iou_heatmap(self, title=None):
        """Create a heatmap visualization for IoU similarity metric"""
        matrix, labels, all_combinations, layer_indices = self._create_matrix_data(
            metric="iou"
        )
        title = title or "Neural Network Layer Similarity - IoU"
        return self._create_heatmap_figure(
            matrix, labels, layer_indices, "iou", title, colorscale=["white", "green"]
        )

    def create_pearson_heatmap(self, title=None):
        """Create a heatmap visualization for Pearson correlation similarity metric"""
        matrix, labels, all_combinations, layer_indices = self._create_matrix_data(
            metric="pearson_r"
        )
        title = title or "Neural Network Layer Similarity - Pearson Correlation"
        return self._create_heatmap_figure(
            matrix,
            labels,
            layer_indices,
            "pearson_r",
            title,
            colorscale=["red", "white", "green"],
            zmax=1,
            zmin=-1,
        )

    def create_heatmap(self, metric="iou", title=None):
        """Create the similarity heatmap with layer grouping (legacy method for backward compatibility)"""
        if metric == "iou":
            return self.create_iou_heatmap(title)
        elif metric == "pearson_r":
            return self.create_pearson_heatmap(title)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'iou' or 'pearson_r'")


def compute_similarity_metrics(task_df):
    matching_columns = [
        col
        for col in task_df.columns
        if col.endswith("dataset_row_id_token_id_act_val")
    ]

    cols = [
        "index",
        "layer",
        *matching_columns,
    ]

    result = []

    for (index1, layer1, *dataset_row_id_token_id_act_val_list1), (
        index2,
        layer2,
        *dataset_row_id_token_id_act_val_list2,
    ) in combinations(task_df[cols].itertuples(index=False), 2):

        total_intersection = 0
        total_union = 0

        act_vals1 = []
        act_vals2 = []

        for dataset_row_id_token_id_act_val1, dataset_row_id_token_id_act_val2 in zip(
            dataset_row_id_token_id_act_val_list1, dataset_row_id_token_id_act_val_list2
        ):
            dataset_row_id_token_id_act_val1_set = {
                row[:2] for row in dataset_row_id_token_id_act_val1
            }
            dataset_row_id_token_id_act_val2_set = {
                row[:2] for row in dataset_row_id_token_id_act_val2
            }

            intersection = (
                dataset_row_id_token_id_act_val1_set
                & dataset_row_id_token_id_act_val2_set
            )
            union = (
                dataset_row_id_token_id_act_val1_set
                | dataset_row_id_token_id_act_val2_set
            )

            total_intersection += len(intersection)
            total_union += len(union)

            sorted_intersection = sorted(intersection)

            if len(sorted_intersection) == 0:
                continue

            iter_sorted_intersection = iter(sorted_intersection)
            current_example_id, current_token_id = next(
                iter_sorted_intersection, (None, None)
            )

            for (
                example_id1,
                curret_token_id1,
                act_val1,
            ) in dataset_row_id_token_id_act_val1:
                if current_example_id == None or current_token_id == None:
                    break

                if (
                    example_id1 == current_example_id
                    and curret_token_id1 == current_token_id
                ):
                    act_vals1.append(act_val1)
                    current_example_id, current_token_id = next(
                        iter_sorted_intersection, (None, None)
                    )

            iter_sorted_intersection = iter(sorted_intersection)
            current_example_id, current_token_id = next(
                iter_sorted_intersection, (None, None)
            )

            for (
                example_id2,
                curret_token_id2,
                act_val2,
            ) in dataset_row_id_token_id_act_val2:
                if current_example_id == None or current_token_id == None:
                    break

                if (
                    example_id2 == current_example_id
                    and curret_token_id2 == current_token_id
                ):
                    act_vals2.append(act_val2)
                    current_example_id, current_token_id = next(
                        iter_sorted_intersection, (None, None)
                    )

        pearson_r, _ = pearsonr(act_vals1, act_vals2) if len(act_vals1) > 1 else (0, 0)

        result.append(
            {
                "index1": index1,
                "layer1": layer1,
                "index2": index2,
                "layer2": layer2,
                "iou": total_intersection / total_union if total_union > 0 else 0,
                "pearson_r": pearson_r,
            }
        )

    final_result = pd.DataFrame(result)

    return final_result


def plot_features_similarity(
    lape_result,
    layers: list[str],
    output_dir: Path,
    task_configs: dict,
    start_index: int = None,
    end_index: int = None,
):
    sorted_lang = lape_result["sorted_lang"]

    for lang in tqdm(sorted_lang[start_index:end_index], desc="Processing languages"):
        lang_index = sorted_lang.index(lang)
        task_df = load_task_df(lape_result, lang, lang_index, layers, task_configs)
        similarity_df = compute_similarity_metrics(task_df)
        heatmap_gen = SimilarityHeatmapGenerator(similarity_df)

        # Create IoU heatmap
        fig_iou = heatmap_gen.create_iou_heatmap(
            title="Neural Network Layer Similarity - IoU"
        )

        output_path = output_dir / "iou" / f"{lang}.html"

        os.makedirs(output_path.parent, exist_ok=True)

        fig_iou.write_html(
            output_path,
            include_plotlyjs="cdn",
        )

        save_image(output_path, fig_iou)

        # Create Pearson correlation heatmap
        fig_pearson = heatmap_gen.create_pearson_heatmap(
            title="Neural Network Layer Similarity - Pearson Correlation"
        )
        output_path = output_dir / "pearson" / f"{lang}.html"
        os.makedirs(output_path.parent, exist_ok=True)
        fig_pearson.write_html(
            output_path,
            include_plotlyjs="cdn",
        )
        save_image(output_path, fig_pearson)


def calculate_correlations(df, score_col):
    valid_data = df.dropna(subset=["entropy", score_col])
    if len(valid_data) < 2:
        return None, None, None, None

    pearson_corr, pearson_pval = pearsonr(valid_data["entropy"], valid_data[score_col])

    return pearson_corr, pearson_pval


def create_scatter_plot(df_subset, metric, score_type, color):
    pearson_corr, _ = calculate_correlations(df_subset, metric)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_subset["entropy"],
            y=df_subset[metric],
            mode="markers",
            marker=dict(color=color, opacity=0.6, size=6),
            name=f"{score_type.title()} Data",
            hovertemplate=f"<b>{metric.title()}</b><br>"
            + "Entropy: %{x}<br>"
            + f"{metric.title()}: %{{y}}<br>"
            + "Lang: %{customdata[0]}<br>"
            + "Layer: %{customdata[1]}<extra></extra>",
            customdata=df_subset[["lang", "layer"]].values,
        )
    )

    # Add trend line if correlation exists
    if pearson_corr is not None:
        x_range = np.linspace(
            df_subset["entropy"].min(), df_subset["entropy"].max(), 100
        )
        slope = pearson_corr * (df_subset[metric].std() / df_subset["entropy"].std())
        intercept = df_subset[metric].mean() - slope * df_subset["entropy"].mean()
        y_trend = slope * x_range + intercept

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_trend,
                mode="lines",
                line=dict(color="black", dash="dash", width=2),
                name="Trend Line",
                hoverinfo="skip",
            )
        )

    # Create correlation text for annotation
    corr_text = ""
    if pearson_corr is not None:
        corr_text = f"Pearson r = {pearson_corr:.3f}"

    title_text = f"{metric.title()} vs Entropy - {score_type.title()} Score Type"
    if corr_text:
        subtitle_text = (
            corr_text.replace("<br>", " | ").replace("<i>", "").replace("</i>", "")
        )
        full_title = f"{title_text}<br><sub>{subtitle_text}</sub>"
    else:
        full_title = title_text

    fig.update_layout(
        title=dict(
            text=full_title,
            x=0.5,
            font=dict(size=16),
        ),
        xaxis_title="Entropy",
        yaxis_title=metric.title(),
        template="plotly_white",
        showlegend=True,
    )

    return fig


def plot_sae_features_entropy_score_correlation(sae_features_info, output_dir: Path):
    score_metrics = ["precision", "recall", "f1_score", "accuracy"]
    color_map = {
        "detection": "#2194e6",
        "fuzz": "#e42d2d",
    }

    graphs = {}

    for score_type in ["detection", "fuzz"]:
        df_filtered = sae_features_info[
            sae_features_info["score_type"] == score_type
        ].copy()

        for metric in score_metrics:
            fig = create_scatter_plot(
                df_filtered, metric, score_type, color_map[score_type]
            )

            graph_key = f"{score_type}_{metric}"
            graphs[graph_key] = fig

            output_path = output_dir / score_type / f"{metric}_vs_entropy.html"
            os.makedirs(output_path.parent, exist_ok=True)

            fig.write_html(
                output_path,
                include_plotlyjs="cdn",
            )

            save_image(output_path, fig, title_keep=True)


def plot_intersection_heatmap(lang_to_count_final_indicies, output_path: Path):
    languages = list(lang_to_count_final_indicies.keys())
    n_languages = len(languages)

    # Create intersection matrix
    intersection_matrix = np.zeros((n_languages, n_languages))

    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i == j:
                # Self-intersection is the size of the set itself
                intersection_size = len(lang_to_count_final_indicies[lang1])
            else:
                shared_indices = lang_to_count_final_indicies[lang1].intersection(
                    lang_to_count_final_indicies[lang2]
                )
                intersection_size = len(shared_indices)

            intersection_matrix[i, j] = intersection_size

    # Convert to ISO codes and sort according to x_label_sort
    languages_labels = [lang_choices_to_iso639_1[lang] for lang in languages]

    label_sort = [
        "en",
        "de",
        "fr",
        "it",
        "pt",
        "hi",
        "es",
        "th",
        "bg",
        "ru",
        "tr",
        "vi",
        "ja",
        "ko",
        "zh",
    ]

    # Create sorting indices based on x_label_sort order
    sort_indices = []
    for target_lang in label_sort:
        for i, lang_label in enumerate(languages_labels):
            if lang_label == target_lang:
                sort_indices.append(i)
                break

    # Add any remaining languages not in x_label_sort
    for i, lang_label in enumerate(languages_labels):
        if i not in sort_indices:
            sort_indices.append(i)

    # Reorder matrix and labels
    sorted_matrix = intersection_matrix[np.ix_(sort_indices, sort_indices)]
    sorted_labels = [languages_labels[i] for i in sort_indices]

    # Reverse y-axis order to match label_sort from top to bottom
    y_sorted_labels = sorted_labels[::-1]
    sorted_matrix = np.flipud(sorted_matrix)

    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=sorted_matrix,
            x=sorted_labels,
            y=y_sorted_labels,
            colorscale="Blues",
            text=sorted_matrix.astype(int),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Intersection Count"),
            hovertemplate="%{y} ∩ %{x}: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Language-Shared Features Intersection Count Matrix",
        title_x=0.5,
        xaxis=dict(tickmode="linear", side="top"),  # Move x-axis labels to top
        yaxis=dict(
            tickmode="linear",
        ),
    )

    fig.update_xaxes(tickfont=dict(size=16), title_text=None)
    fig.update_yaxes(tickfont=dict(size=16), title_text=None)

    os.makedirs(output_path.parent, exist_ok=True)

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
    )

    save_image(output_path, fig)


def plot_iou_heatmap(lang_to_count_final_indicies, output_path: Path):
    languages = list(lang_to_count_final_indicies.keys())
    n_languages = len(languages)

    # Create Jaccard similarity matrix
    iou_matrix = np.zeros((n_languages, n_languages))

    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i == j:
                iou = 1.0  # Self-similarity is 1
            else:
                shared_indices = lang_to_count_final_indicies[lang1].intersection(
                    lang_to_count_final_indicies[lang2]
                )
                union_indices = lang_to_count_final_indicies[lang1].union(
                    lang_to_count_final_indicies[lang2]
                )
                iou = (
                    len(shared_indices) / len(union_indices)
                    if len(union_indices) > 0
                    else 0
                )

            iou_matrix[i, j] = iou

    # Convert to ISO codes and sort according to x_label_sort
    languages_labels = [lang_choices_to_iso639_1[lang] for lang in languages]

    label_sort = [
        "en",
        "de",
        "fr",
        "it",
        "pt",
        "hi",
        "es",
        "th",
        "bg",
        "ru",
        "tr",
        "vi",
        "ja",
        "ko",
        "zh",
    ]

    # Create sorting indices based on x_label_sort order
    sort_indices = []
    for target_lang in label_sort:
        for i, lang_label in enumerate(languages_labels):
            if lang_label == target_lang:
                sort_indices.append(i)
                break

    # Add any remaining languages not in x_label_sort
    for i, lang_label in enumerate(languages_labels):
        if i not in sort_indices:
            sort_indices.append(i)

    # Reorder matrix and labels
    sorted_matrix = iou_matrix[np.ix_(sort_indices, sort_indices)]
    sorted_labels = [languages_labels[i] for i in sort_indices]

    # Reverse y-axis order to match label_sort from top to bottom
    y_sorted_labels = sorted_labels[::-1]
    sorted_matrix = np.flipud(sorted_matrix)

    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=sorted_matrix,
            x=sorted_labels,
            y=y_sorted_labels,
            colorscale="Reds",
            text=np.round(sorted_matrix, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Jaccard Similarity"),
            hovertemplate="%{y} ∩ %{x}: %{z:.3f}<extra></extra>",
            zmin=0,
            zmax=1,
        )
    )

    fig.update_layout(
        title="Language-Shared Features Intersection over Union Matrix",
        title_x=0.5,
        xaxis=dict(tickmode="linear", side="top"),  # Move x-axis labels to top
        yaxis=dict(
            tickmode="linear",
        ),
    )

    fig.update_xaxes(tickfont=dict(size=16), title_text=None)
    fig.update_yaxes(tickfont=dict(size=16), title_text=None)

    os.makedirs(output_path.parent, exist_ok=True)

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
    )

    save_image(output_path, fig)


def plot_shared_count_bar_chart(
    data_dict, output_path, title="Shared-Feature Distribution by Layer Index"
):
    if not isinstance(data_dict, dict):
        print("Error: Input data must be a Python dictionary.")
        return

    if not data_dict:
        print("Error: Input dictionary is empty.")
        return

    traces = []
    # Sort shared_counts to ensure consistent trace order and legend colors
    sorted_shared_counts = sorted(data_dict.keys())

    for shared_count in sorted_shared_counts:
        features_set = data_dict[shared_count]

        if not features_set:
            print(f"Warning: No features for shared_count {shared_count}. Skipping.")
            continue

        # Extract layer indices from the tuples
        current_layer_indices = []
        for item in features_set:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                # item[0] is the layer_index
                current_layer_indices.append(item[0])
            else:
                print(
                    f"Warning: Invalid feature item format '{item}' for shared_count {shared_count}. Skipping item."
                )

        if not current_layer_indices:
            print(
                f"Warning: No valid layer indices found for shared_count {shared_count}. Skipping."
            )
            continue

        # Count occurrences of each layer index
        layer_index_counts = Counter(current_layer_indices)

        if not layer_index_counts:
            continue  # No data to plot for this shared_count

        # Sort layer indices numerically
        sorted_layer_indices = sorted(layer_index_counts.keys())
        y_counts = [layer_index_counts[idx] for idx in sorted_layer_indices]

        traces.append(
            go.Bar(
                name=f"{shared_count}",
                x=sorted_layer_indices,  # Use integers directly
                y=y_counts,
                hovertemplate=(
                    f"<b>Shared Count: {shared_count}</b><br>"
                    "Layer Index: %{x}<br>"
                    "Feature Count: %{y}<extra></extra>"
                ),
            )
        )

    if not traces:
        print("No plottable data found. No chart will be generated.")
        return

    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis=dict(
            title="Layer",
            type="linear",
            tickmode="linear",
        ),
        yaxis_title="Count",
        legend_title_text="Shared Counts",
        template="plotly_white",
        showlegend=True,
    )

    os.makedirs(output_path.parent, exist_ok=True)

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
    )

    save_image(output_path, fig)


def metric_to_radar_fig(categories, scores_dict: dict[str, list[float]], title: str):
    fig = go.Figure()

    categories_closed = categories + [categories[0]]

    color_options = [
        ("rgb(0, 123, 255)", "rgba(0, 123, 255, 0.1)"),
        ("rgb(255, 99, 71)", "rgba(255, 99, 71, 0.1)"),
        ("rgb(60, 179, 113)", "rgba(60, 179, 113, 0.1)"),
        ("rgb(255, 165, 0)", "rgba(255, 165, 0, 0.1)"),
        ("rgb(75, 0, 130)", "rgba(75, 0, 130, 0.1)"),
        ("rgb(238, 130, 238)", "rgba(238, 130, 238, 0.1)"),
        ("rgb(255, 20, 147)", "rgba(255, 20, 147, 0.1)"),
    ]

    for (score_name, scores), (line_color, fillcolor) in zip(
        scores_dict.items(), color_options
    ):
        scores_closed = scores + [scores[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=scores_closed,
                theta=categories_closed,
                fill="toself",
                line_color=line_color,
                fillcolor=fillcolor,
                mode="lines",
                name=score_name,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.8, 1.0],
                tickvals=[0.8, 0.85, 0.9, 0.95, 1.0],
                ticktext=["0.80", "0.85", "0.90", "0.95", "1.00"],
                tickfont=dict(size=10),
                gridcolor="lightgray",
                linecolor="black",
            ),
            bgcolor="white",
            angularaxis=dict(linecolor="black", gridcolor="lightgray"),
        ),
        title=dict(text=title.title(), x=0.5, y=0.5, font=dict(size=16)),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.165, xanchor="center", x=0.5
        ),
    )

    return fig


def plot_fastext_vs_sae_metrics(
    saes_based_lid_metric,
    fastext_classifier_metric,
    neurons_based_lid_metric,
    output_dir: Path,
):
    categories = [
        lang_choices_to_iso639_1[class_name]
        for class_name in fastext_classifier_metric["classes"]
    ]

    f1_fig = metric_to_radar_fig(
        categories,
        {
            "FastText": fastext_classifier_metric["f1"],
            "SAE-Based LID": saes_based_lid_metric["f1"],
            "Neuron-Based LID": neurons_based_lid_metric["f1"],
        },
        "F1",
    )

    prec_fig = metric_to_radar_fig(
        categories,
        {
            "FastText": fastext_classifier_metric["precision"],
            "SAE-Based LID": saes_based_lid_metric["precision"],
            "Neuron-Based LID": neurons_based_lid_metric["precision"],
        },
        "Prec.",
    )

    rec_fig = metric_to_radar_fig(
        categories,
        {
            "FastText": fastext_classifier_metric["recall"],
            "SAE-Based LID": saes_based_lid_metric["recall"],
            "Neuron-Based LID": neurons_based_lid_metric["recall"],
        },
        "Rec.",
    )

    # Create subplots with 1 row and 3 columns for the three metrics
    combined_fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "polar"}, {"type": "polar"}, {"type": "polar"}]],
        subplot_titles=["F1", "Prec.", "Rec."],
    )

    # To update the y placement of subplot titles, modify the annotation y value:
    for ann in combined_fig.layout.annotations:
        ann.y = 0.675

    # Add F1 score traces (show legend)
    for trace in f1_fig.data:
        combined_fig.add_trace(trace, row=1, col=1)

    # Add Precision traces (hide legend)
    for trace in prec_fig.data:
        trace.showlegend = False
        combined_fig.add_trace(trace, row=1, col=2)

    # Add Recall traces (hide legend)
    for trace in rec_fig.data:
        trace.showlegend = False
        combined_fig.add_trace(trace, row=1, col=3)

    # Update polar axes to match the original styling
    for i in range(1, 4):
        combined_fig.update_layout(
            {
                f"polar{i}": dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0.4, 1.0],
                        tickvals=[0.4, 0.55, 0.7, 0.85, 1.0],
                        tickfont=dict(size=11),
                        gridcolor="lightgray",
                        linecolor="black",
                    ),
                    bgcolor="white",
                    angularaxis=dict(
                        linecolor="black",
                        gridcolor="lightgray",
                        tickfont=dict(size=14),  # Set category axis font size here
                    ),
                ),
            }
        )

    combined_fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=0.08, xanchor="center", x=0.5),
        width=750,
    )

    # Save the combined figure
    os.makedirs(output_dir.parent, exist_ok=True)

    output_path = output_dir / "classifier_comparisons.pdf"
    save_image(output_path, combined_fig)


def plot_entropy_distribution(sae_features_info, output_dir: Path):
    df_copy = sae_features_info.copy()
    df_copy["lang"] = df_copy["lang"].apply(lambda x: lang_choices_to_iso639_1[x])

    fig = px.histogram(
        df_copy,
        x="entropy",
        nbins=30,
        title="Distribution of Feature Entropy",
        labels={"entropy": "Entropy", "count": "Count", "y": "Count"},
        color="lang",
        color_discrete_map=language_colors,
    )
    fig.update_layout(yaxis_title="Count")
    fig.update_layout(
        bargap=0.1,
        plot_bgcolor="white",
        xaxis=dict(
            tickmode="linear",
            dtick=0.2,
        ),
    )

    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = output_dir / "distribution.html"

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
    )

    save_image(output_path, fig)
