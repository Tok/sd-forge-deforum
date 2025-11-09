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


# ============================================================================
# Orbit Test Visualization (Depth Warping with Rotation Factor Sweep)
# ============================================================================

def create_orbit_metrics_plot(results: List[Dict[str, Any]]) -> plt.Figure:
    """Create line plot showing drift metrics vs rotation factor.

    Args:
        results: List of orbit test result dictionaries

    Returns:
        Matplotlib figure object
    """
    if not results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No results yet', ha='center', va='center')
        ax.set_title('Orbit Test Metrics')
        return fig

    # Group by aspect ratio
    aspect_ratios = sorted(list(set(r['aspect_ratio'] for r in results)))

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for i, aspect in enumerate(aspect_ratios):
        # Filter results for this aspect ratio
        aspect_results = [r for r in results if r['aspect_ratio'] == aspect]
        aspect_results = sorted(aspect_results, key=lambda r: r['rotation_factor'])

        rotation_factors = [r['rotation_factor'] for r in aspect_results]
        max_drifts = [r['max_drift'] for r in aspect_results]
        overall_scores = [r['overall_score'] for r in aspect_results]

        # Color from slopcore gradient
        color = [SLOPCORE_2, SLOPCORE_4, SLOPCORE_6][i % 3]

        # Plot drift (lower is better)
        ax1.plot(rotation_factors, max_drifts, 'o-', label=f'Aspect {aspect:.2f}',
                color=color, linewidth=2, markersize=6)

        # Plot overall score (higher is better)
        ax2.plot(rotation_factors, overall_scores, 'o-', label=f'Aspect {aspect:.2f}',
                color=color, linewidth=2, markersize=6)

    # Format drift plot
    ax1.set_xlabel('Rotation Factor')
    ax1.set_ylabel('Max Drift (pixels)')
    ax1.set_title('Subject Position Drift vs Rotation Factor')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # More negative on right

    # Format score plot
    ax2.set_xlabel('Rotation Factor')
    ax2.set_ylabel('Overall Score (0-100)')
    ax2.set_title('Overall Quality Score vs Rotation Factor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()  # More negative on right
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    return fig


def create_orbit_heatmap(results: List[Dict[str, Any]], metric: str = 'overall_score') -> plt.Figure:
    """Create heatmap showing metric across aspect ratios and rotation factors.

    Args:
        results: List of orbit test result dictionaries
        metric: Metric to visualize (overall_score, max_drift, etc.)

    Returns:
        Matplotlib figure object
    """
    if not results:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No results yet', ha='center', va='center')
        ax.set_title('Orbit Parameter Heatmap')
        return fig

    # Group by aspect ratio and rotation factor
    aspect_ratios = sorted(list(set(r['aspect_ratio'] for r in results)))
    rotation_factors = sorted(list(set(r['rotation_factor'] for r in results)), reverse=True)

    # Create matrix
    matrix = np.zeros((len(aspect_ratios), len(rotation_factors)))

    for result in results:
        i = aspect_ratios.index(result['aspect_ratio'])
        j = rotation_factors.index(result['rotation_factor'])
        matrix[i, j] = result[metric]

    # Create heatmap with slopcore gradient colormap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use appropriate colormap based on metric
    if metric == 'max_drift' or metric == 'avg_drift' or metric == 'drift_rate':
        # For drift metrics, invert colormap (lower is better)
        cmap = SLOPCORE_CMAP.reversed()
        vmin, vmax = 0, np.max(matrix) if np.max(matrix) > 0 else 100
    else:
        # For score metrics, normal colormap (higher is better)
        cmap = SLOPCORE_CMAP
        vmin, vmax = 0, 100

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    # Set ticks
    ax.set_xticks(np.arange(len(rotation_factors)))
    ax.set_yticks(np.arange(len(aspect_ratios)))
    ax.set_xticklabels([f'{f:.1f}' for f in rotation_factors])
    ax.set_yticklabels([f'{a:.2f}' for a in aspect_ratios])

    # Labels
    ax.set_xlabel('Rotation Factor')
    ax.set_ylabel('Aspect Ratio')

    metric_labels = {
        'overall_score': 'Overall Quality Score',
        'max_drift': 'Maximum Drift (pixels)',
        'avg_drift': 'Average Drift (pixels)',
        'drift_rate': 'Drift Rate (pixels/frame)',
        'temporal_consistency': 'Temporal Consistency',
    }
    ax.set_title(f'Orbit Parameter Heatmap: {metric_labels.get(metric, metric)}')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score' if 'score' in metric else 'Drift (px)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(aspect_ratios)):
        for j in range(len(rotation_factors)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    return fig


def find_best_orbit_configuration(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the best orbit parameter configuration from results.

    Best is defined as highest overall_score (lowest drift + highest temporal).

    Args:
        results: List of orbit test result dictionaries

    Returns:
        Best configuration dict or None if no results
    """
    if not results:
        return None

    return max(results, key=lambda r: r['overall_score'])


def generate_orbit_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from orbit test results.

    Args:
        results: List of orbit test result dictionaries

    Returns:
        Dictionary of summary statistics
    """
    if not results:
        return {
            'total_tests': 0,
            'best_overall_score': 0,
            'avg_overall_score': 0,
            'best_config': None,
            'avg_max_drift': 0,
            'min_max_drift': 0,
        }

    overall_scores = [r['overall_score'] for r in results]
    max_drifts = [r['max_drift'] for r in results]
    best_config = find_best_orbit_configuration(results)

    return {
        'total_tests': len(results),
        'best_overall_score': max(overall_scores),
        'avg_overall_score': np.mean(overall_scores),
        'std_overall_score': np.std(overall_scores),
        'best_config': best_config,
        'worst_overall_score': min(overall_scores),
        'avg_max_drift': np.mean(max_drifts),
        'min_max_drift': min(max_drifts),
        'max_max_drift': max(max_drifts),
    }
