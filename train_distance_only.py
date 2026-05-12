"""
train_distance_only.py
======================
Versione del training PPO con reward basata SOLO sulla distanza.
- Tutti i pesi relativi a copertura (SINR/QoS) sono azzerati.
- I risultati vengono salvati in: results_distance_only/
"""

import os
import sys
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import argparse
import time
import math
import csv

sys.setrecursionlimit(5000)

from env.route_planner_env import RoutePlannerDRL
from enviroment.sumo_env import set_sumo_env


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
    """RLlib stores custom metrics in different places across versions."""
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


def main():
    parser = argparse.ArgumentParser(description="Train DRL Agent - Distance Only Reward")
    parser.add_argument("--gui",     action="store_true", help="Launch SUMO GUI")
    parser.add_argument("--steps",   type=int, default=400000, help="Max training timesteps")
    parser.add_argument("--gpu",     action="store_true", help="Enable GPU usage")
    parser.add_argument("--workers", type=int, default=0, help="Number of parallel workers")
    parser.add_argument("--resume",  type=str, default="", help="Path to a full RLlib checkpoint to resume")
    parser.add_argument("--no-clear-csv", action="store_true", help="Append to the existing CSV instead of clearing it")
    parser.add_argument("--checkpoint-freq", type=int, default=5, help="Save a full training checkpoint every N iterations")
    args = parser.parse_args()

    repo_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(repo_path, "results_distance_only")
    os.makedirs(output_dir, exist_ok=True)

    sumo_cfg = os.path.join(repo_path, "sumo", "Esch-Belval.train.sumocfg")
    sumo_net = os.path.join(repo_path, "sumo", "Esch-Belval.net.xml")

    env_config = {
        "sumo_cfg_file": sumo_cfg,
        "sumo_net_file": sumo_net,
        "gui": args.gui,
        "startEdge":    "361567664",
        "endEdge":      "291922340#2",
        "revStartEdge": "-361567664",
        "csv_file":     os.path.join(output_dir, "reward_distance_only.csv"),
        "checkpoint_dir": os.path.join(output_dir, "checkpoint"),
        # ── DISTANCE ONLY ──────────────────────────────────────────────
        "TARGET_ACQUIRED_WEIGHT":    1000,   # +1000 se raggiunge la meta
        "ILLEGAL_ACTION_WEIGHT":      500,   # -500 se va fuori mappa
        "DISTANCE_REWARD_WEIGHT":      15,   # ±15 × Δdist/100m (unico segnale)
        # I seguenti pesi sono TUTTI azzerati: l'agente ignora copertura e QoS
        "COVERAGE_REWARD_WEIGHT":       0,
        "QOS_REWARD_WEIGHT":            0,
        "MIN_COVERAGE_REWARD_WEIGHT":   0,
        "COVERAGE_TRESHOLD":           15,   # soglia tecnica (non usata nel reward)
        # Stabilizzazione su mappa complessa
        "LOOP_PENALTY_WEIGHT":          10,
        "INVALID_ACTION_WEIGHT":         5,
        "STEP_PENALTY":                1.0,
        "MAX_EPISODE_STEPS":           160,
        "verbose":                   False,
    }

    # ── Ray & PPO Config ──────────────────────────────────────────────────
    ray.init(ignore_reinit_error=True)

    num_gpus = 1 if args.gpu else 0
    print(f"GPU: {'Enabled' if num_gpus > 0 else 'Disabled'}")

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
    try:
        config.training(sgd_minibatch_size=512, num_sgd_iter=10)
    except TypeError:
        try:
            config.training(minibatch_size=512, num_epochs=10)
        except TypeError:
            pass
    try:
        config.api_stack(enable_rl_module_and_learner=False,
                         enable_env_runner_and_connector_v2=False)
    except Exception:
        pass

    algo = config.build()
    if args.resume:
        print(f"Restoring full RLlib checkpoint from: {args.resume}")
        algo.restore(args.resume)

    # ── Clear CSV ─────────────────────────────────────────────────────────
    csv_path = env_config["csv_file"]
    append_csv = args.no_clear_csv or bool(args.resume)
    if not append_csv or not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timesteps",
                "episode_reward_mean",
                "episode_reward_min",
                "episode_reward_max",
                "episode_len_mean",
                "success_rate",
                "out_of_bound_rate",
                "truncated_rate",
                "revisited_edges_mean",
            ])
    print(f"Logging rewards to: {csv_path}")

    # ── Early Stopping ────────────────────────────────────────────────────
    best_reward    = -float('inf')
    best_ckpt_dir  = env_config["checkpoint_dir"] + "_best"
    full_ckpt_root = os.path.join(output_dir, "rllib_checkpoints")
    patience       = 30
    patience_counter = 0
    min_delta      = 10.0

    # ── ANSI Colors ───────────────────────────────────────────────────────
    C_BLUE   = "\033[94m"
    C_GREEN  = "\033[92m"
    C_YELLOW = "\033[93m"
    C_RED    = "\033[91m"
    C_BOLD   = "\033[1m"
    C_END    = "\033[0m"

    total_timesteps = 0
    train_iteration = 0
    start_time = time.time()

    print("\n" + "="*60)
    print(f"🚀 TRAINING: DISTANCE ONLY")
    print(f"📍 Start: 361567664 | End: 291922340#2")
    print(f"📂 Output: {output_dir}")
    print("="*60 + "\n")

    while total_timesteps < args.steps:
        try:
            res = algo.train()
            train_iteration += 1
            total_timesteps = res["timesteps_total"]

            metrics       = res.get("env_runners", res)
            current_reward = metrics.get('episode_reward_mean', float('nan'))
            min_reward    = metrics.get('episode_reward_min', float('nan'))
            max_reward    = metrics.get('episode_reward_max', float('nan'))
            ep_total      = res.get('episodes_total', 0)
            ep_len        = metrics.get('episode_len_mean', 0)
            custom        = get_custom_metrics(res, metrics)
            success_rate  = custom.get("goal_reached_mean", 0) * 100
            oob_rate      = custom.get("out_of_bound_mean", 0) * 100
            trunc_rate    = custom.get("truncated_mean", 0) * 100
            revisits_mean = custom.get("revisited_edges_mean", 0)

            # Learner stats
            entropy     = 0
            policy_loss = 0
            if "learner" in res and "default_policy" in res["learner"]:
                ls = res["learner"]["default_policy"]["learner_stats"]
                entropy     = ls.get("entropy", 0)
                policy_loss = ls.get("policy_loss", 0)

            timestamp = time.strftime("%H:%M:%S")
            progress  = (total_timesteps / args.steps) * 100
            bar_len   = 20
            filled    = int(bar_len * progress / 100)
            bar       = "█" * filled + "░" * (bar_len - filled)

            rew_color = C_GREEN if current_reward > best_reward else C_YELLOW
            if not math.isnan(current_reward) and current_reward < 0:
                rew_color = C_RED

            print(f"\n{C_BOLD}{C_BLUE}--- Distance-Only Training [{timestamp}] ---{C_END}")
            print(f"{C_BOLD}Progress:{C_END} [{bar}] {progress:5.1f}% | Steps: {total_timesteps:,}")
            print(f"{C_BOLD}Episodes:{C_END} {ep_total:,} | Avg Len: {ep_len:.1f}")
            print(f"{C_BOLD}Reward:  {rew_color}{current_reward:8.2f}{C_END} (Min: {min_reward:7.1f} | Max: {max_reward:7.1f})")
            print(f"{C_BOLD}Route:   {success_rate:6.1f}% Goal | {oob_rate:6.1f}% OOB | {trunc_rate:6.1f}% Trunc | Revisits: {revisits_mean:.2f}")
            print(f"{C_BOLD}Learner:{C_END} Entropy: {entropy:.3f} | Loss: {policy_loss:.4f}")

            patience_str = f"{patience_counter}/{patience}"
            if patience_counter > 0:
                print(f"{C_YELLOW}⚠️  No improvement for {patience_str} (Best: {best_reward:.2f}){C_END}")
            else:
                print(f"{C_GREEN}⭐ New Best! (Best: {best_reward:.2f}){C_END}")

            # ── CSV ──────────────────────────────────────────────────────
            try:
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        total_timesteps,
                        current_reward,
                        min_reward,
                        max_reward,
                        ep_len,
                        success_rate,
                        oob_rate,
                        trunc_rate,
                        revisits_mean,
                    ])
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as fe:
                print(f"❌ ERRORE CSV: {fe}")

            # ── Early Stopping ───────────────────────────────────────────
            if current_reward is None or math.isnan(current_reward):
                print(f"{C_YELLOW}⚠️  Reward NaN: nessun episodio completato. Ignoro.{C_END}")
            elif current_reward > best_reward + min_delta:
                best_reward = current_reward
                patience_counter = 0
                try:
                    os.makedirs(best_ckpt_dir, exist_ok=True)
                    policy = algo.get_policy()
                    policy.export_checkpoint(best_ckpt_dir)
                    full_best = save_training_checkpoint(algo, full_ckpt_root, "best")
                    print(f"{C_GREEN}Full RLlib checkpoint: {full_best}{C_END}")
                    print(f"{C_GREEN}💾 Miglior checkpoint! {best_reward:.2f} → {best_ckpt_dir}{C_END}")
                except Exception as ce:
                    print(f"⚠️ Impossibile salvare checkpoint: {ce}")
            else:
                patience_counter += 1

            if args.checkpoint_freq > 0 and train_iteration % args.checkpoint_freq == 0:
                try:
                    latest = save_training_checkpoint(algo, full_ckpt_root, "latest")
                    print(f"{C_GREEN}Full checkpoint periodico: {latest}{C_END}")
                except Exception as ce:
                    print(f"Impossibile salvare checkpoint periodico: {ce}")

            if patience_counter >= patience:
                print(f"\n{C_BOLD}{C_RED}✅ [CONVERGENZA] {patience} iter senza miglioramento. Stop.{C_END}")
                break

        except KeyboardInterrupt:
            print("\nInterruzione richiesta. Salvo checkpoint completo prima di uscire...")
            break

        except Exception as e:
            print(f"❌ Errore: {e}")
            import traceback
            traceback.print_exc()
            break

    print("\nTraining finished. Saving final checkpoint...")
    sys.setrecursionlimit(5000)
    try:
        os.makedirs(env_config["checkpoint_dir"], exist_ok=True)
        policy = algo.get_policy()
        policy.export_checkpoint(env_config["checkpoint_dir"])
        final_full = save_training_checkpoint(algo, full_ckpt_root, "final")
        print(f"Checkpoint finale salvato in {env_config['checkpoint_dir']}")
        print(f"Full RLlib checkpoint finale: {final_full}")
    except Exception as e:
        print(f"⚠️ Impossibile salvare checkpoint finale: {e}")


if __name__ == "__main__":
    main()
