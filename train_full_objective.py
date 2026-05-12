"""
train_full_objective.py
=======================
PPO training for the full routing objective:
distance + coverage/SINR + QoS.

Outputs are written under results_full_objective/.
"""

import argparse
import csv
import math
import os
import sys
import time

import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig

sys.setrecursionlimit(5000)

from env.route_planner_env import RoutePlannerDRL


class RouteMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, episode, **kwargs):
        info = episode.last_info_for()
        if not info:
            return
        for key in (
            "goal_reached",
            "out_of_bound",
            "truncated",
            "route_steps",
            "revisited_edges",
            "avg_sinr",
            "avg_qos",
        ):
            if key in info:
                episode.custom_metrics[key] = info[key]


def get_custom_metrics(result, metrics):
    for candidate in (
        result.get("custom_metrics"),
        metrics.get("custom_metrics") if isinstance(metrics, dict) else None,
        result.get("env_runners", {}).get("custom_metrics")
        if isinstance(result.get("env_runners"), dict)
        else None,
    ):
        if candidate:
            return candidate
    return {}


def save_training_checkpoint(algo, checkpoint_root, label):
    checkpoint_dir = os.path.join(checkpoint_root, label)
    os.makedirs(checkpoint_dir, exist_ok=True)
    saved = algo.save(checkpoint_dir)
    return getattr(saved, "checkpoint", saved)


def add_legacy_api_guard(config):
    try:
        config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    except Exception:
        pass


def add_minibatch_config(config):
    try:
        config.training(sgd_minibatch_size=512, num_sgd_iter=10)
    except TypeError:
        try:
            config.training(minibatch_size=512, num_epochs=10)
        except TypeError:
            pass


def write_csv_header(csv_path, append_csv):
    if append_csv and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timesteps",
                "episode_reward_mean",
                "episode_reward_min",
                "episode_reward_max",
                "episode_len_mean",
                "success_rate",
                "out_of_bound_rate",
                "truncated_rate",
                "revisited_edges_mean",
                "avg_sinr",
                "avg_qos",
                "coverage_scale",
                "qos_scale",
                "min_coverage_scale",
            ]
        )


def reward_color(value, best_value, colors):
    if value is None or math.isnan(value):
        return colors["yellow"]
    if value > best_value:
        return colors["green"]
    if value < 0:
        return colors["red"]
    return colors["yellow"]


def curriculum_scales(total_timesteps, warmup_steps, ramp_steps):
    if warmup_steps <= 0 and ramp_steps <= 0:
        return 1.0, 1.0, 1.0
    if total_timesteps < warmup_steps:
        return 0.0, 0.0, 0.0
    if ramp_steps <= 0:
        return 1.0, 1.0, 1.0
    progress = min(1.0, max(0.0, (total_timesteps - warmup_steps) / ramp_steps))
    return progress, progress, progress * progress


def apply_curriculum_scales(algo, coverage_scale, qos_scale, min_cov_scale):
    def update_env(env):
        env.coverage_weight_scale = coverage_scale
        env.qos_weight_scale = qos_scale
        env.min_coverage_weight_scale = min_cov_scale

    try:
        algo.workers.local_worker().foreach_env(update_env)
    except Exception:
        pass
    try:
        algo.workers.foreach_worker(lambda worker: worker.foreach_env(update_env))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Train PPO route planner - full objective")
    parser.add_argument("--mode", type=str, default="full", help="Deprecated; this trainer always uses the full objective")
    parser.add_argument("--gui", action="store_true", help="Launch SUMO GUI")
    parser.add_argument("--steps", type=int, default=800000, help="Max training timesteps")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU usage")
    parser.add_argument("--workers", type=int, default=0, help="Number of parallel workers")
    parser.add_argument("--resume", type=str, default="", help="Path to a full RLlib checkpoint to resume")
    parser.add_argument("--no-clear-csv", action="store_true", help="Append to the existing CSV")
    parser.add_argument("--checkpoint-freq", type=int, default=5, help="Save full checkpoint every N iterations")
    parser.add_argument("--distance-weight", type=float, default=15.0)
    parser.add_argument("--coverage-weight", type=float, default=5.0)
    parser.add_argument("--qos-weight", type=float, default=10.0)
    parser.add_argument("--min-coverage-weight", type=float, default=120.0)
    parser.add_argument("--coverage-threshold", type=float, default=15.0)
    parser.add_argument("--coverage-warmup-steps", type=int, default=300000)
    parser.add_argument("--coverage-ramp-steps", type=int, default=300000)
    args = parser.parse_args()
    if args.mode != "full":
        print(f"Warning: --mode {args.mode} is deprecated here; train_full_objective.py always uses the full objective.")

    repo_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(repo_path, "results_full_objective")
    os.makedirs(output_dir, exist_ok=True)

    sumo_cfg = os.path.join(repo_path, "sumo", "Esch-Belval.train.sumocfg")
    sumo_net = os.path.join(repo_path, "sumo", "Esch-Belval.net.xml")

    env_config = {
        "sumo_cfg_file": sumo_cfg,
        "sumo_net_file": sumo_net,
        "gui": args.gui,
        "startEdge": "361567664",
        "endEdge": "291922340#2",
        "revStartEdge": "-361567664",
        "csv_file": os.path.join(output_dir, "reward_full_objective.csv"),
        "checkpoint_dir": os.path.join(output_dir, "checkpoint"),
        "TARGET_ACQUIRED_WEIGHT": 1000,
        "ILLEGAL_ACTION_WEIGHT": 500,
        "DISTANCE_REWARD_WEIGHT": args.distance_weight,
        "COVERAGE_REWARD_WEIGHT": args.coverage_weight,
        "QOS_REWARD_WEIGHT": args.qos_weight,
        "MIN_COVERAGE_REWARD_WEIGHT": args.min_coverage_weight,
        "COVERAGE_TRESHOLD": args.coverage_threshold,
        "COVERAGE_WEIGHT_SCALE": 0.0,
        "QOS_WEIGHT_SCALE": 0.0,
        "MIN_COVERAGE_WEIGHT_SCALE": 0.0,
        "LOOP_PENALTY_WEIGHT": 10,
        "INVALID_ACTION_WEIGHT": 5,
        "STEP_PENALTY": 1.0,
        "MAX_EPISODE_STEPS": 160,
        "verbose": False,
    }

    ray.init(ignore_reinit_error=True)

    num_gpus = 1 if args.gpu else 0
    print(f"GPU: {'Enabled' if num_gpus > 0 else 'Disabled'}")
    print("Full objective: distance + coverage/SINR + QoS")
    if args.workers > 1:
        print("Warning: each worker loads coverage/QoS models. Use high worker counts only with enough RAM.")

    config = (
        PPOConfig()
        .environment(RoutePlannerDRL, env_config=env_config)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .callbacks(RouteMetricsCallback)
        .env_runners(
            num_env_runners=args.workers,
            rollout_fragment_length=500,
        )
        .training(
            train_batch_size=8000,
            lr=1e-4,
            gamma=0.995,
            lambda_=0.95,
            clip_param=0.1,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
            },
        )
    )
    add_minibatch_config(config)
    add_legacy_api_guard(config)

    algo = config.build()
    if args.resume:
        print(f"Restoring full RLlib checkpoint from: {args.resume}")
        algo.restore(args.resume)

    csv_path = env_config["csv_file"]
    append_csv = args.no_clear_csv or bool(args.resume)
    write_csv_header(csv_path, append_csv)
    print(f"Logging rewards to: {csv_path}")

    best_reward = -float("inf")
    best_ckpt_dir = env_config["checkpoint_dir"] + "_best"
    full_ckpt_root = os.path.join(output_dir, "rllib_checkpoints")
    patience = 40
    patience_counter = 0
    min_delta = 10.0

    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "bold": "\033[1m",
        "end": "\033[0m",
    }

    total_timesteps = 0
    train_iteration = 0
    start_time = time.time()

    print("\n" + "=" * 60)
    print("TRAINING: FULL OBJECTIVE")
    print("Start: 361567664 | End: 291922340#2")
    print(f"Output: {output_dir}")
    print("=" * 60 + "\n")

    try:
        while total_timesteps < args.steps:
            res = algo.train()
            train_iteration += 1
            total_timesteps = res["timesteps_total"]
            coverage_scale, qos_scale, min_cov_scale = curriculum_scales(
                total_timesteps,
                args.coverage_warmup_steps,
                args.coverage_ramp_steps,
            )
            apply_curriculum_scales(algo, coverage_scale, qos_scale, min_cov_scale)

            metrics = res.get("env_runners", res)
            current_reward = metrics.get("episode_reward_mean", float("nan"))
            min_reward = metrics.get("episode_reward_min", float("nan"))
            max_reward = metrics.get("episode_reward_max", float("nan"))
            ep_total = res.get("episodes_total", 0)
            ep_len = metrics.get("episode_len_mean", 0)

            custom = get_custom_metrics(res, metrics)
            success_rate = custom.get("goal_reached_mean", 0) * 100
            oob_rate = custom.get("out_of_bound_mean", 0) * 100
            trunc_rate = custom.get("truncated_mean", 0) * 100
            revisits_mean = custom.get("revisited_edges_mean", 0)
            avg_sinr = custom.get("avg_sinr_mean", 0)
            avg_qos = custom.get("avg_qos_mean", 0)

            entropy = 0
            policy_loss = 0
            if "learner" in res and "default_policy" in res["learner"]:
                learner_stats = res["learner"]["default_policy"]["learner_stats"]
                entropy = learner_stats.get("entropy", 0)
                policy_loss = learner_stats.get("policy_loss", 0)

            timestamp = time.strftime("%H:%M:%S")
            progress = (total_timesteps / args.steps) * 100

            bar_len = 20
            filled = int(bar_len * min(max(progress, 0.0), 100.0) / 100.0)
            bar = "#" * filled + "." * (bar_len - filled)
            rew_color = reward_color(current_reward, best_reward, colors)
            success_color = colors["green"] if success_rate >= 50 else colors["yellow"]
            risk_color = colors["red"] if oob_rate > 40 or trunc_rate > 40 else colors["yellow"]

            print(f"\n{colors['bold']}{colors['blue']}--- Full-Objective Training [{timestamp}] ---{colors['end']}")
            print(f"{colors['bold']}Progress:{colors['end']} [{bar}] {progress:5.1f}% | Steps: {total_timesteps:,} | Episodes: {ep_total:,}")
            print(
                f"{colors['bold']}Reward:  {rew_color}{current_reward:8.2f}{colors['end']} "
                f"(Min: {min_reward:7.1f} | Max: {max_reward:7.1f})"
            )
            print(
                f"{colors['bold']}Route:   {success_color}{success_rate:6.1f}% Goal{colors['end']} | "
                f"{risk_color}{oob_rate:6.1f}% OOB | {trunc_rate:6.1f}% Trunc{colors['end']} | "
                f"Revisits: {revisits_mean:.2f}"
            )
            print(f"{colors['bold']}Signal:{colors['end']}  SINR: {avg_sinr:7.2f} | QoS: {avg_qos:7.2f} | Len: {ep_len:.1f}")
            print(
                f"{colors['bold']}Weights:{colors['end']} coverage x{coverage_scale:.2f} | "
                f"qos x{qos_scale:.2f} | min-coverage x{min_cov_scale:.2f}"
            )
            print(f"{colors['bold']}Learner:{colors['end']} Entropy: {entropy:.3f} | Loss: {policy_loss:.4f}")
            if patience_counter > 0:
                print(f"{colors['yellow']}No improvement: {patience_counter}/{patience} (Best: {best_reward:.2f}){colors['end']}")

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        total_timesteps,
                        current_reward,
                        min_reward,
                        max_reward,
                        ep_len,
                        success_rate,
                        oob_rate,
                        trunc_rate,
                        revisits_mean,
                        avg_sinr,
                        avg_qos,
                        coverage_scale,
                        qos_scale,
                        min_cov_scale,
                    ]
                )
                f.flush()
                os.fsync(f.fileno())

            if current_reward is None or math.isnan(current_reward):
                print("Reward NaN: no completed episodes in this batch. Ignoring for early stopping.")
            elif current_reward > best_reward + min_delta:
                best_reward = current_reward
                patience_counter = 0
                try:
                    os.makedirs(best_ckpt_dir, exist_ok=True)
                    algo.get_policy().export_checkpoint(best_ckpt_dir)
                    full_best = save_training_checkpoint(algo, full_ckpt_root, "best")
                    print(f"Best policy checkpoint: {best_ckpt_dir}")
                    print(f"Best full RLlib checkpoint: {full_best}")
                except Exception as ce:
                    print(f"Could not save best checkpoint: {ce}")
            else:
                patience_counter += 1

            if args.checkpoint_freq > 0 and train_iteration % args.checkpoint_freq == 0:
                try:
                    latest = save_training_checkpoint(algo, full_ckpt_root, "latest")
                    print(f"Periodic full checkpoint: {latest}")
                except Exception as ce:
                    print(f"Could not save periodic checkpoint: {ce}")

            if patience_counter >= patience:
                print(f"No improvement for {patience} iterations. Stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted. Saving final checkpoint before exit...")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback

        traceback.print_exc()

    print("\nTraining finished. Saving final checkpoints...")
    try:
        os.makedirs(env_config["checkpoint_dir"], exist_ok=True)
        algo.get_policy().export_checkpoint(env_config["checkpoint_dir"])
        final_full = save_training_checkpoint(algo, full_ckpt_root, "final")
        print(f"Policy checkpoint: {env_config['checkpoint_dir']}")
        print(f"Full RLlib checkpoint: {final_full}")
    except Exception as e:
        print(f"Could not save final checkpoint: {e}")

    elapsed = time.time() - start_time
    print(f"Elapsed seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
