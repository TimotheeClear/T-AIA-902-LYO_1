from textwrap import indent
from typing import Any, Callable
from uu import Error
from markov.Training import AgentMetrics
import matplotlib.pyplot as plt
import numpy as np

def validateType(type : str):
    if type != "step" and type != "episode" and type != "model_readiness":
        raise Error(f"`type` argument must be either 'step' or 'episode' or `model_readiness`.") 

def barMetricComparison(
    labels  : list,
    heights : list,
    xlabel  : str | None    = None,
    ylabel  : str | None    = None,
    title   : str | None    = None,
):
    plt.bar(
        x       = labels,
        height  = heights
    )

    if xlabel is not None:
        plt.xlabel(xlabel)
        
    if ylabel is not None:
        plt.ylabel(ylabel)
        
    if title is not None:
        plt.title(title)

    plt.show()
    
def BarAggregatedMetricsComparison(
    agent_metrics   : dict[str, AgentMetrics],
    type            : str,
    agg             : Callable[[list], Any],
    filter          : list[str] | None  = None,
):
    validateType(type)
    flat_metrics = FlattenMetrics(type, agent_metrics)

    fig, ax             = plt.subplots()
    for metric_name, metric_agent_values in flat_metrics.items(): 
        
        if filter is not None and metric_name not in filter:
            continue

        labels  = []
        heights = []
        for agent_name, values in metric_agent_values.items():
            labels.append(agent_name)
            heights.append(agg(values))

        barMetricComparison(
            labels  = labels,
            heights = heights,
            xlabel  = "Agents",
            ylabel  = metric_name,
            title   = f"Aggregated {metric_name}",
        )


    

def plotMetricComparison(
    values  : dict[str, list],
    xlabel  : str | None    = None,
    ylabel  : str | None    = None,
    title   : str | None    = None,
    step    : int           = 1
):
    fig, ax             = plt.subplots()

    max_length = max(len(agent_values) for agent_values in values.values())
    for agent in values:
        current_length = len(values[agent])
        if current_length < max_length:
            values[agent] += [float('nan')] * (max_length - current_length)
            
    x = range(max_length)
    
    for agent, agent_values in values.items():
        ax.plot(x[::step], agent_values[::step], label=agent)

    plt.legend()

    if xlabel is not None:
        plt.xlabel(xlabel)
        
    if ylabel is not None:
        plt.ylabel(ylabel)
        
    if title is not None:
        plt.title(title)

    plt.show()

def PlotMetricsComparison(
    agent_metrics   : dict[str, AgentMetrics],
    type            : str,
    filter          : list[str] | None  = None,
    step            : int               = 1
):
    validateType(type)
    flat_metrics = FlattenMetrics(type, agent_metrics)
    
    for metric_name, metric_agent_values in flat_metrics.items(): 
        
        if filter is not None and metric_name not in filter:
            continue
        
        plotMetricComparison(
            values  = metric_agent_values,
            title   = metric_name,
            xlabel  = type,
            ylabel  = metric_name,
            step    = step
        )

def FlattenMetrics(
    type : str, 
    agent_metrics   : dict[str, AgentMetrics]
) -> dict[str,dict[str,list]]:
    result = {}
    
    for agent_name, metrics_obj in agent_metrics.items():
        type_metrics = metrics_obj.step_metrics if type == "step" else metrics_obj.episode_metrics if type == "episode" else metrics_obj.model_readiness_metrics

        for metric_name, metric_func in type_metrics.items():
            if metric_name not in result:
                result[metric_name] = {}

            result[metric_name][agent_name] = metrics_obj.results[type][metric_name]
    
    return result
        
# Broken Axes

import numpy as np
import matplotlib.pyplot as plt


def plotMetricComparisonBrokenAxes(values: dict[str, list], xlabel: str | None = None, ylabel: str | None = None, title: str | None = None, step: int = 1, top_focus: int = 10, mid_focus: int = 500):
    # Calculate overall min and max values
    all_values = np.concatenate([np.array(values[agent]) for agent in values])
    min_val, max_val = np.min(all_values), np.max(all_values)

    # Define the thresholds for breaking the y-axis
    top_break = max_val - top_focus
    mid_break = max_val - top_focus - mid_focus

    # Create three subplots with shared x-axis
    fig, (ax3, ax2, ax) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    # Set the limits for each axis
    ax.set_ylim(min_val, mid_break)  # Bottom plot for the lowest values
    ax2.set_ylim(mid_break, top_break)  # Middle plot for the mid-range values
    ax3.set_ylim(top_break, max_val)  # Top plot for the top range

    # Hide the spines between subplots
    ax.spines['bottom'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['top'].set_visible(True)
    ax3.spines['top'].set_visible(True)
    
    ax.xaxis.tick_top()
    ax.xaxis.tick_bottom()

    ax2.xaxis.tick_top()
    ax2.xaxis.tick_bottom()

    ax3.xaxis.tick_bottom()
    
    ax.xaxis.set_tick_params(which='both', labelbottom=True,)
    ax2.xaxis.set_tick_params(which='both', labelbottom=True)
    ax3.xaxis.set_tick_params(which='both', labelbottom=True)

    # Draw diagonal lines on the breaks
    d = .005  # Diagonal line size
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax3.transAxes)
    ax3.plot((-d, +d), (-d, +d), **kwargs)
    ax3.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    max_length = max(len(agent_values) for agent_values in values.values())
    x = range(0, max_length, 1)

    # Plot data on all three axes and annotate last value
    for agent, agent_values in values.items():
        if len(agent_values) < max_length:
            agent_values += [float('nan')] * (max_length - len(agent_values))
        line, = ax.plot(x, agent_values, label=agent)
        ax2.plot(x, agent_values, label=agent)
        ax3.plot(x, agent_values, label=agent)
        # Annotate the last value
        for a in [ax, ax2, ax3]:
            a.annotate(f'{agent_values[-1]:.2f}', xy=(x[-1], agent_values[-1]), xytext=(10, 0), 
                       textcoords='offset points', ha='right', va='bottom',
                       color=line.get_color())


    ax3.legend(loc='upper left')

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)
        ax3.set_ylabel(ylabel)
    if title:
        ax3.set_title(title)

    # Ensure x-ticks on the bottom plot align with the data indices properly
    ax.set_xticks(x[::step])
    ax2.set_xticks(x[::step])
    ax3.set_xticks(x[::step])
    x_labels = [((i+1) *10) for i in x]
    ax.set_xticklabels(x_labels[::step])
    ax2.set_xticklabels(x_labels[::step])
    ax3.set_xticklabels(x_labels[::step])
    # grind
    ax.grid(True)
    ax2.grid(True)
    ax3.grid(True)


    plt.tight_layout()
    plt.show()

# Adjust `plotMetricComparisonBrokenAxes` usage as needed in your code to apply these changes.

# Sample usage within your existing structure
def PlotMetricsComparisonBrokenAxes(
    agent_metrics   : dict[str, AgentMetrics],
    type            : str,
    filter          : list[str] | None  = None,
    step            : int               = 1,
    top_focus       : int               = 10,
    mid_focus       : int               = 500
):
    validateType(type)
    flat_metrics = FlattenMetrics(type, agent_metrics)
    
    for metric_name, metric_agent_values in flat_metrics.items(): 
        if filter is not None and metric_name not in filter:
            continue
        
        plotMetricComparisonBrokenAxes(
            values  = metric_agent_values,
            title   = metric_name,
            xlabel  = type,
            ylabel  = metric_name,
            step    = step,
            top_focus = top_focus,
            mid_focus = mid_focus
        )
