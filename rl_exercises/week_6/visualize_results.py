from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rliable import library as rly
from rliable import metrics, plot_utils


class ResultsVisualizer:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.eval_file = self.results_dir / "evaluation_results.csv"

        # Load data
        try:
            if self.eval_file.exists():
                self.eval_df = pd.read_csv(self.eval_file)
                print(f"Loaded evaluation data: {len(self.eval_df)} rows")
            else:
                print(f"Warning: {self.eval_file} not found")
        except Exception as e:
            print(f"Error loading data: {e}")

        # Output directory for plots
        self.output_dir = self.results_dir / "plots"
        self.output_dir.mkdir(exist_ok=True)

        print("Results visualizer initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")

    def prepare_rliable_data(self):
        if self.eval_df is None:
            print("No results data available")
            return {}

        score_dict = {}
        algorithms = self.eval_df["algorithm"].unique()

        for algorithm in algorithms:
            alg_data = self.eval_df[self.eval_df["algorithm"] == algorithm]
            unique_seeds = alg_data["seed"].unique()
            scores = []
            for seed in unique_seeds:
                scores.append(alg_data[alg_data["seed"] == seed]["return"].values)
            scores = np.array(scores)
            score_dict[algorithm.upper()] = scores[
                :, np.newaxis, :
            ]  # Add new axis for compatibility

        return score_dict

    def plot_with_rliable(self):
        score_dict = self.prepare_rliable_data()
        if not score_dict:
            return

        steps = np.array(self.eval_df["step"].unique())

        fig, ax = plt.subplots(figsize=(15, 12))
        fig.suptitle("Actor-Critic Baselines", fontsize=16, fontweight="bold")

        def aggregate_func(scores):
            return np.array(
                [
                    metrics.aggregate_mean(scores[..., frame])
                    for frame in range(scores.shape[-1])
                ]
            )

        algorithms = list(score_dict.keys())
        aggregate_scores, aggregate_cis = rly.get_interval_estimates(
            score_dict, aggregate_func, reps=1000
        )

        plot_utils.plot_sample_efficiency_curve(
            steps,
            aggregate_scores,
            aggregate_cis,
            algorithms=algorithms,
            xlabel="Steps",
            ylabel="Mean Returns",
            ax=ax,
        )

        color_dict = dict(zip(algorithms, sns.color_palette("colorblind")))

        fake_patches = [
            mpatches.Patch(color=color_dict[alg], alpha=1) for alg in algorithms
        ]
        fig.legend(
            fake_patches,
            algorithms,
            loc="upper center",
            fancybox=True,
            ncol=len(algorithms) // 2,
            fontsize="x-large",
            bbox_to_anchor=(0.8, 1.05),
        )

        plt.tight_layout()
        plt.show()
        plot_path = self.output_dir / "actor_critic_baselines.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"RLiable analysis saved to {plot_path}")


def main():
    result_dir = "outputs/experiments/20250601_231452"

    visualizer = ResultsVisualizer(result_dir)

    # Create plots
    visualizer.plot_with_rliable()

    print(f"\nVisualization completed! Plots saved to: {visualizer.output_dir}")


if __name__ == "__main__":
    main()
