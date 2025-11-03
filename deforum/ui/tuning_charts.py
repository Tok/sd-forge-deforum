"""Chart generation for tuning metrics visualization.

This module provides functions to generate matplotlib charts for
displaying tuning test results with slopcore gradient aesthetics.
"""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from typing import List, Dict, Any, Optional

# Use non-interactive backend for server-side generation
matplotlib.use('Agg')

# Authentic BLANK BANSHEE 0 gradient colors (pipetted from album cover)
SLOPCORE_1 = '#5606FF'  # Deep purple-blue (album top)
SLOPCORE_2 = '#4C21FF'  # Purple-blue
SLOPCORE_3 = '#413CFF'  # Blue-purple
SLOPCORE_4 = '#3757FF'  # Mid blue
SLOPCORE_5 = '#2C71FE'  # Blue
SLOPCORE_6 = '#228CFE'  # Bright blue
SLOPCORE_7 = '#17A7FE'  # Cyan (album bottom)

# Create custom slopcore colormap for heatmaps (dark purple → bright cyan)
SLOPCORE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'slopcore',
    [SLOPCORE_1, SLOPCORE_2, SLOPCORE_3, SLOPCORE_4, SLOPCORE_5, SLOPCORE_6, SLOPCORE_7]
)


def create_metrics_plot(results: List[Dict[str, Any]]) -> plt.Figure:
    """Create line plot showing quality metrics across configurations.

    Args:
        results: List of test result dictionaries

    Returns:
        Matplotlib figure object
    """
    if not results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No results yet', ha='center', va='center')
        ax.set_title('Quality Metrics')
        return fig

    # Extract data
    configs = [
        f"S{r['steps']}_N{r['normal_strength']:.2f}_KF{r['keyframe_strength']:.2f}"
        for r in results
    ]
    final_color = [r['final_color_score'] for r in results]
    avg_temporal = [r['avg_temporal_consistency'] for r in results]
    overall = [r['overall_score'] for r in results]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(configs))
    width = 0.25

    # Plot bars with slopcore gradient colors
    ax.bar(x - width, final_color, width, label='Final Color', alpha=0.8, color=SLOPCORE_3)
    ax.bar(x, avg_temporal, width, label='Avg Temporal', alpha=0.8, color=SLOPCORE_5)
    ax.bar(x + width, overall, width, label='Overall Score', alpha=0.8, color=SLOPCORE_7)

    # Formatting
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score (0-100)')
    ax.set_title('Quality Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return fig


def create_heatmap_plot(results: List[Dict[str, Any]], metric: str = 'overall_score') -> plt.Figure:
    """Create heatmap showing metric across parameter combinations.

    Args:
        results: List of test result dictionaries
        metric: Metric to visualize (overall_score, final_color_score, etc.)

    Returns:
        Matplotlib figure object
    """
    if not results:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No results yet', ha='center', va='center')
        ax.set_title('Parameter Heatmap')
        return fig

    # Group by strength values
    normal_strengths = sorted(list(set(r['normal_strength'] for r in results)))
    kf_strengths = sorted(list(set(r['keyframe_strength'] for r in results)))

    # Create matrix
    matrix = np.zeros((len(kf_strengths), len(normal_strengths)))

    for result in results:
        i = kf_strengths.index(result['keyframe_strength'])
        j = normal_strengths.index(result['normal_strength'])
        matrix[i, j] = result[metric]

    # Create heatmap with slopcore gradient colormap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap=SLOPCORE_CMAP, aspect='auto', vmin=0, vmax=100)

    # Set ticks
    ax.set_xticks(np.arange(len(normal_strengths)))
    ax.set_yticks(np.arange(len(kf_strengths)))
    ax.set_xticklabels([f'{s:.2f}' for s in normal_strengths])
    ax.set_yticklabels([f'{s:.2f}' for s in kf_strengths])

    # Labels
    ax.set_xlabel('Normal Strength')
    ax.set_ylabel('Keyframe Strength')

    metric_labels = {
        'overall_score': 'Overall Quality Score',
        'final_color_score': 'Final Color Score',
        'avg_temporal_consistency': 'Avg Temporal Consistency',
        'iterations_completed': 'Iterations Completed',
    }
    ax.set_title(f'Parameter Heatmap: {metric_labels.get(metric, metric)}')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(kf_strengths)):
        for j in range(len(normal_strengths)):
            text = ax.text(j, i, f'{matrix[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    return fig


def create_degradation_plot(results: List[Dict[str, Any]]) -> plt.Figure:
    """Create plot showing degradation rates across configurations.

    Args:
        results: List of test result dictionaries

    Returns:
        Matplotlib figure object
    """
    if not results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No results yet', ha='center', va='center')
        ax.set_title('Degradation Rates')
        return fig

    # Extract data
    configs = [
        f"S{r['steps']}_N{r['normal_strength']:.2f}_KF{r['keyframe_strength']:.2f}"
        for r in results
    ]
    degradation_rates = [r['degradation_rate'] for r in results]
    iterations = [r['iterations_completed'] for r in results]

    # Create plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(configs))
    width = 0.35

    # Plot degradation rates (lower is better) - use darker purple for warning
    color = SLOPCORE_7
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Degradation Rate (%/iteration)', color=color)
    ax1.bar(x - width/2, degradation_rates, width, label='Degradation Rate',
            color=color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')

    # Plot iterations (higher is better) - use brighter blue for positive metric
    ax2 = ax1.twinx()
    color = SLOPCORE_2
    ax2.set_ylabel('Iterations Completed', color=color)
    ax2.bar(x + width/2, iterations, width, label='Iterations',
            color=color, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_title('Degradation Rate vs Iterations Completed')
    ax1.grid(axis='y', alpha=0.3)

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    return fig


def find_best_configuration(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the best parameter configuration from results.

    Best is defined as highest overall_score.

    Args:
        results: List of test result dictionaries

    Returns:
        Best configuration dict or None if no results
    """
    if not results:
        return None

    return max(results, key=lambda r: r['overall_score'])


def generate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from results.

    Args:
        results: List of test result dictionaries

    Returns:
        Dictionary of summary statistics
    """
    if not results:
        return {
            'total_tests': 0,
            'best_overall_score': 0,
            'avg_overall_score': 0,
            'best_config': None,
        }

    overall_scores = [r['overall_score'] for r in results]
    best_config = find_best_configuration(results)

    return {
        'total_tests': len(results),
        'best_overall_score': max(overall_scores),
        'avg_overall_score': np.mean(overall_scores),
        'std_overall_score': np.std(overall_scores),
        'best_config': best_config,
        'worst_overall_score': min(overall_scores),
    }
