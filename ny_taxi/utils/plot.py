import os
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from typing import List
from matplotlib.figure import Figure


def create_pie_chart(
    sizes: np.ndarray,
    labels: List[str],
    plot_title: str,
    legend_title: str,
    show_percent: bool = True,
    sort_legend: bool = True,
) -> Figure:
    colors = sns.color_palette("pastel")[: len(sizes)]

    fig = plt.figure(figsize=(6, 6))
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.axis("equal")
    plt.title(plot_title)

    if show_percent:
        percent = 100.0 * sizes / sizes.sum()
        labels = [f"{i} - {j:1.2f}% ({k})" for i, j, k in zip(labels, percent, sizes)]
    else:
        labels = [f"{i} - {j}" for i, j in zip(labels, sizes)]

    sort_legend = True
    if sort_legend:
        patches, labels, dummy = zip(
            *sorted(
                zip(patches, labels, sizes), key=lambda labels: labels[2], reverse=True
            )
        )

    plt.legend(patches, labels, loc="best", title=legend_title, fontsize=8)
    return fig


def create_bar_chart(
    data: np.ndarray,
    labels: List[str],
    plot_title: str,
) -> Figure:
    colors = sns.color_palette("pastel")[: len(data)]
    fig = plt.figure(figsize=(6, 6))
    x_labels = np.arange(len(data))
    plt.title(plot_title)
    plt.grid()
    plt.bar(x_labels, data, label=labels, color=colors)
    plt.xticks(x_labels, labels, rotation="vertical")
    return fig
