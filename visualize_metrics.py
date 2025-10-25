import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

from metrics import MusicGenerationMetrics


class MusicVisualizer:
    def __init__(self, output_dir="generation_plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_music_timeseries(self, music, title="Generated Music", filename="music_timeseries.png"):
        """Plot music as a piano roll timeseries"""
        fig, ax = plt.subplots(figsize=(15, 8))

        colors = plt.cm.Set3(np.linspace(0, 1, len(music.tracks)))

        for i, track in enumerate(music.tracks):
            if len(track.notes) == 0:
                continue

            color = colors[i]
            instrument_name = f"Instr_{track.program}"

            for note in track.notes:
                # Create rectangle for each note
                rect = patches.Rectangle(
                    (note.time, note.pitch - 0.4),  # x, y (time, pitch)
                    note.duration, 0.8,  # width, height
                    linewidth=1, edgecolor='black',
                    facecolor=color, alpha=0.7,
                    label=instrument_name if i == 0 else ""
                )
                ax.add_patch(rect)

        # Customize plot
        ax.set_xlabel('Time (ticks)')
        ax.set_ylabel('Pitch')
        ax.set_title(f'{title}\nTotal Notes: {sum(len(track.notes) for track in music.tracks)}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 128)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.values(), title='Instruments')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_velocity_timeseries(self, music, title="Velocity Over Time", filename="velocity_timeseries.png"):
        """Plot velocity as a timeseries"""
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, track in enumerate(music.tracks):
            if len(track.notes) == 0:
                continue

            times = [note.time for note in track.notes]
            velocities = [note.velocity for note in track.notes]

            ax.scatter(times, velocities, alpha=0.7, label=f'Instr_{track.program}', s=30)

        ax.set_xlabel('Time (ticks)')
        ax.set_ylabel('Velocity')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()


class MetricsPlotter:
    def __init__(self, output_dir="metrics_plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_single_metrics(self, metrics, title="Generation Metrics", filename="single_metrics.png"):
        """Plot metrics for a single generation"""
        # Categorize metrics
        structural_metrics = {
            'ssm_score': 'Structural Coherence',
            'scale_consistency': 'Scale Consistency'
        }

        rhythmic_metrics = {
            'ioi_variance': 'Rhythmic Consistency',
            'gross_rhythm_consistency': 'Gross Rhythm Consistency'
        }

        conditional_metrics = {
            'instrument_usage_ratio': 'Instrument Usage Ratio',
            'pitch_range': 'Pitch Range'
        }

        style_metrics = {
            'style_score': 'Style Adherence',
            'pitch_entropy': 'Pitch Entropy'
        }

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Plot each category
        categories = [
            (structural_metrics, 'Structural Metrics', 'blue', axes[0]),
            (rhythmic_metrics, 'Rhythmic Metrics', 'green', axes[1]),
            (conditional_metrics, 'Conditional Metrics', 'orange', axes[2]),
            (style_metrics, 'Style Metrics', 'red', axes[3])
        ]

        for metric_dict, category_title, color, ax in categories:
            available_metrics = {k: v for k, v in metric_dict.items() if k in metrics}
            if available_metrics:
                names = list(available_metrics.values())
                values = [metrics[k] for k in available_metrics.keys()]

                bars = ax.bar(names, values, color=color, alpha=0.7)
                ax.set_title(category_title)
                ax.set_ylabel('Score')

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=9)

                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_comparison_metrics(self, metrics_list, labels, title="Metrics Comparison",
                                filename="comparison_metrics.png"):
        """Plot metrics comparison across multiple generations"""
        if not metrics_list:
            return

        # Get all available metrics
        all_metrics = set()
        for metrics in metrics_list:
            all_metrics.update(metrics.keys())
        all_metrics = sorted(all_metrics)

        # Create radar plot for comparison
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

        # Normalize metrics for radar plot
        normalized_metrics = []
        for metrics in metrics_list:
            normalized = {}
            for metric in all_metrics:
                if metric in metrics:
                    # Simple normalization (you might want more sophisticated normalization)
                    normalized[metric] = min(metrics[metric] / 1.0, 1.0)  # Assuming max score of 1.0
                else:
                    normalized[metric] = 0.0
            normalized_metrics.append(normalized)

        # Prepare angles
        angles = np.linspace(0, 2 * np.pi, len(all_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Plot each generation
        colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_list)))

        for i, (norm_metrics, label, color) in enumerate(zip(normalized_metrics, labels, colors)):
            values = [norm_metrics[metric] for metric in all_metrics]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color, alpha=0.7)
            ax.fill(angles, values, alpha=0.1, color=color)

        # Add metric names
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_metrics, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=16, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_metrics_timeseries(self, metrics_history, title="Metrics Over Time", filename="metrics_timeseries.png"):
        """Plot metrics evolution over multiple generations"""
        if not metrics_history:
            return

        # Extract metrics over time
        metrics_over_time = {}
        for step_metrics in metrics_history:
            for metric, value in step_metrics.items():
                if metric not in metrics_over_time:
                    metrics_over_time[metric] = []
                metrics_over_time[metric].append(value)

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        metrics_to_plot = ['ssm_score', 'ioi_variance', 'instrument_usage_ratio', 'style_score']

        for i, metric in enumerate(metrics_to_plot):
            if metric in metrics_over_time and len(metrics_over_time[metric]) > 1:
                ax = axes[i]
                steps = range(len(metrics_over_time[metric]))
                ax.plot(steps, metrics_over_time[metric], 'o-', linewidth=2, markersize=4)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel('Generation Step')
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()


class GeneratorMetrics:

    def __init__(self, generator=None, preset_name=None, style=None, instruments_used=None, prompt_tokens=None,
                 num_generations=3):
        self.preset_name = preset_name
        self.style = style
        self.instruments_used = instruments_used
        self.prompt_tokens = prompt_tokens
        self.num_generations = num_generations
        self.generator = generator

    def analyze_and_plot_generation(self, prompt_tokens, output_prefix="generation"):
        """Complete analysis and plotting for a generation"""
        # Initialize plotters
        music_viz = MusicVisualizer()
        metrics_plotter = MetricsPlotter()
        metrics_calculator = MusicGenerationMetrics()

        # Generate music
        print("Generating music...")
        midi_music, tokens = midi_music, _ = self.generator.generate_with_preset(
            preset_name=self.preset_name,
            num_steps=100,
            prompt_tokens=prompt_tokens,
            style=self.style,
            allowed_instruments=self.instruments_used,
            notes_per_chord=3,
            temperature=0.5,
            include_initial=False,
            top_k=10,
            top_p=0.9
        )

        # Plot music timeseries
        print("Plotting music timeseries...")
        music_viz.plot_music_timeseries(
            midi_music,
            title=f"{output_prefix} - Piano Roll",
            filename=f"{output_prefix}_piano_roll.png"
        )

        music_viz.plot_velocity_timeseries(
            midi_music,
            title=f"{output_prefix} - Velocity Distribution",
            filename=f"{output_prefix}_velocity.png"
        )

        # Calculate metrics
        print("Calculating metrics...")
        metrics = metrics_calculator.compute_all_metrics(
            midi_music,
            target_instruments=self.instruments_used
        )

        # Plot metrics
        print("Plotting metrics...")
        metrics_plotter.plot_single_metrics(
            metrics,
            title=f"{output_prefix} - Quality Metrics",
            filename=f"{output_prefix}_metrics.png"
        )

        # Print summary
        print(f"\n📊 {output_prefix} Metrics Summary:")
        print(f"   Structural Coherence: {metrics.get('ssm_score', 0):.3f}")
        print(f"   Rhythmic Consistency: {metrics.get('ioi_variance', 0):.3f}")
        print(f"   Instrument Usage Ratio: {metrics.get('instrument_usage_ratio', 0):.3f}")
        print(f"   Style Adherence: {metrics.get('style_score', 0):.3f}")

        return midi_music, tokens, metrics

    # Multiple generations comparison
    def compare_multiple_generations(self):
        """Compare multiple generations"""
        all_metrics = []
        all_music = []

        for i in range(self.num_generations):
            print(f"\n🎵 Generating sample {i + 1}/{self.num_generations}")
            midi_music, tokens, metrics = self.analyze_and_plot_generation(
                self.prompt_tokens, f"sample_{i + 1}"
            )
            all_metrics.append(metrics)
            all_music.append(midi_music)

        # Plot comparison
        metrics_plotter = MetricsPlotter()
        labels = [f"Sample {i + 1}" for i in range(self.num_generations)]

        metrics_plotter.plot_comparison_metrics(
            all_metrics, labels,
            title="Multiple Generations Comparison",
            filename="generations_comparison.png"
        )

        return all_music, all_metrics
