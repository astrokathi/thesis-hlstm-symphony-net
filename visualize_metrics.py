import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

from generator import MusicGenerator
from metrics import MusicGenerationMetrics
import wav_composer as wc


class MusicVisualizer:
    def __init__(self, output_dir="assets/generation_plots", gen_method=1):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/{gen_method}", exist_ok=True)


    def plot_music_timeseries(self, music, title="Generated Music", filename="music_timeseries.png"):
        """Plot music as a piano roll timeseries"""
        fig, ax = plt.subplots(figsize=(15, 10))

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
        fig, ax = plt.subplots(figsize=(15, 10))

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
    def __init__(self, output_dir="assets/metrics_plots", gen_method=1):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/{gen_method}", exist_ok=True)

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
        fig, ax = plt.subplots(figsize=(15, 10), subplot_kw=dict(projection='polar'))

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

    def __init__(self,
                 generator: MusicGenerator = None,
                 preset_name=None,
                 style=None,
                 instruments_used=None,
                 prompt_tokens=None,
                 num_generations=3,
                 num_steps=100,
                 control_context=[],
                 notes_per_chord=2,
                 temperature=0.7,
                 include_initial=False,
                 top_k=20,
                 top_p=0.9,
                 generation_method=0,
                 seq_length=128
                 ):
        self.preset_name = preset_name
        self.style = style
        self.instruments_used = instruments_used
        self.prompt_tokens = prompt_tokens
        self.num_generations = num_generations
        self.generator = generator
        self.num_steps = num_steps
        self.control_context = control_context
        self.notes_per_chord = notes_per_chord
        self.temperature = temperature
        self.include_initial = include_initial
        self.top_k = top_k
        self.top_p = top_p
        self.tokens = None
        self.generation_method = generation_method
        self.midi_music = None
        self.seq_length = seq_length

    def analyze_and_plot_generation(self, output_prefix="generation"):
        """Complete analysis and plotting for a generation"""
        # Initialize plotters
        music_viz = MusicVisualizer(gen_method=self.generation_method)
        metrics_plotter = MetricsPlotter(gen_method=self.generation_method)
        metrics_calculator = MusicGenerationMetrics()

        # Generate music
        print(f"Generating music and metrics for the method {self.generation_method}")
        if self.generation_method == 1:
            self.midi_music, self.tokens = self.generator.method_1(
                prompt_tokens=self.prompt_tokens,
                style=self.style,
                num_steps=self.num_steps,
                temperature=self.temperature,
                top_k=self.top_k,
                include_initial=self.include_initial,
                allowed_instruments=self.instruments_used
            )
        if self.generation_method == 2:
            self.midi_music, self.tokens = self.generator.method_2(
                prompt_tokens=self.prompt_tokens,
                style=self.style,
                num_steps=self.num_steps,
                temperature=self.temperature,
                top_k=self.top_k,
                include_initial=self.include_initial,
                allowed_instruments=self.instruments_used,
                seq_len=self.seq_length
            )
        if self.generation_method == 3:
            self.midi_music, self.tokens = self.generator.method_3(
                prompt_tokens=self.prompt_tokens,
                style=self.style,
                num_steps=self.num_steps,
                temperature=self.temperature,
                top_k=self.top_k,
                include_initial=self.include_initial,
                allowed_instruments=self.instruments_used,
                notes_per_chord=self.notes_per_chord,
                top_p=self.top_p,
                seq_len=self.seq_length
            )
        if self.generation_method == 4:
            self.midi_music, self.tokens = self.generator.method_4(
                preset_name=self.preset_name,
                prompt_tokens=self.prompt_tokens,
                style=self.style,
                num_steps=self.num_steps,
                temperature=self.temperature,
                top_k=self.top_k,
                include_initial=self.include_initial,
                allowed_instruments=self.instruments_used,
                notes_per_chord=self.notes_per_chord,
                seq_len=self.seq_length,
                top_p=self.top_p,
                control_context=np.array(self.control_context)
            )

        # Plot music timeseries
        print("Plotting music timeseries...")
        music_viz.plot_music_timeseries(
            self.midi_music,
            title=f"{self.generation_method}_{output_prefix} - Piano Roll",
            filename=f"{self.generation_method}/{output_prefix}_piano_roll.png"
        )

        music_viz.plot_velocity_timeseries(
            self.midi_music,
            title=f"{self.generation_method}_{output_prefix} - Velocity Distribution",
            filename=f"{self.generation_method}/{output_prefix}_velocity.png"
        )

        # Calculate metrics
        print("Calculating metrics...")
        metrics = metrics_calculator.compute_all_metrics(
            self.midi_music,
            target_instruments=self.instruments_used
        )

        # Plot metrics
        print("Plotting metrics...")
        metrics_plotter.plot_single_metrics(
            metrics,
            title=f"{self.generation_method}_{output_prefix} - Quality Metrics",
            filename=f"{self.generation_method}/{output_prefix}_metrics.png"
        )

        # Print summary
        print(f"\n📊 {output_prefix} Metrics Summary:")
        print(f"   Structural Coherence: {metrics.get('ssm_score', 0):.3f}")
        print(f"   Rhythmic Consistency: {metrics.get('ioi_variance', 0):.3f}")
        print(f"   Instrument Usage Ratio: {metrics.get('instrument_usage_ratio', 0):.3f}")
        print(f"   Style Adherence: {metrics.get('style_score', 0):.3f}")

        print(f" Generating assets that can be rendered in the HTML")
        os.makedirs(f"assets/mid/{self.generation_method}", exist_ok=True)
        os.makedirs(f"assets/wav/{self.generation_method}", exist_ok=True)
        wc.write_mid_and_wav(
            music=self.midi_music,
            mid_file_path=f"assets/mid/{self.generation_method}/{output_prefix}_song.mid",
            wav_file_path=f"assets/wav/{self.generation_method}/{output_prefix}_song.wav"
        )

        return self.midi_music, self.tokens, metrics

    # Multiple generations comparison
    def compare_multiple_generations(self):
        """Compare multiple generations"""
        all_metrics = []
        all_music = []

        for i in range(self.num_generations):
            print(f"\n🎵 Generating sample {i + 1}/{self.num_generations}")
            midi_music, tokens, metrics = self.analyze_and_plot_generation(
                f"sample_{i + 1}"
            )
            all_metrics.append(metrics)
            all_music.append(midi_music)

        # Plot comparison
        metrics_plotter = MetricsPlotter()
        labels = [f"Sample {i + 1}" for i in range(self.num_generations)]

        metrics_plotter.plot_comparison_metrics(
            all_metrics, labels,
            title="Multiple Generations Comparison",
            filename=f"{self.generation_method}/generations_comparison.png"
        )

        return all_music, all_metrics
