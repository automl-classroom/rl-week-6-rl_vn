import csv
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from rl_exercises.week_6.actor_critic import ActorCriticAgent, set_seed

# Configuration
algorithms = ["none", "avg", "value", "gae"]
seeds = [0, 1, 2]
total_steps = 800000
eval_interval = 10000
eval_episodes = 10

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("outputs") / "experiments" / timestamp
output_dir.mkdir(parents=True, exist_ok=True)

eval_results_file = output_dir / "evaluation_results.csv"

print("Running baseline comparison...")
print(f"Algorithms: {algorithms}")
print(f"Seeds: {seeds}")
print(f"Output: {output_dir}")

all_eval_data = []

for algorithm in algorithms:
    print(f"\n--- Training {algorithm.upper()} ---")

    for seed in seeds:
        print(f"Running {algorithm} with seed {seed}...")

        # Create environment and agent
        env = gym.make("LunarLander-v3")
        set_seed(env, seed)
        agent = ActorCriticAgent(
            env,
            lr_actor=5e-3,
            lr_critic=1e-2,
            gamma=0.99,
            gae_lambda=0.95,
            hidden_size=128,
            baseline_type=algorithm,
            seed=seed,
        )

        # Training loop
        eval_env = gym.make("LunarLander-v3")
        step_count = 0

        while step_count < total_steps:
            state, _ = env.reset()
            done = False
            trajectory = []

            while not done and step_count < total_steps:
                action, logp = agent.predict_action(state)
                next_state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                trajectory.append(
                    (state, action, float(reward), next_state, done, logp)
                )
                state = next_state
                step_count += 1

                # Evaluation
                if step_count % eval_interval == 0:
                    mean_return, std_return = agent.evaluate(
                        eval_env, num_episodes=eval_episodes
                    )

                    all_eval_data.append(
                        {
                            "algorithm": algorithm,
                            "seed": seed,
                            "step": step_count,
                            "return": mean_return,
                            "return_std": std_return,
                        }
                    )

                    print(
                        f"  Step {step_count:6d} | Return {mean_return:6.1f} ± {std_return:4.1f}"
                    )

            agent.update_agent(trajectory)

        env.close()
        eval_env.close()

# Save results to CSV
with open(eval_results_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["algorithm", "seed", "step", "return", "return_std"]
    )
    writer.writeheader()
    writer.writerows(all_eval_data)

print(f"\nResults saved to: {output_dir}")
print(f"  - {eval_results_file}")
