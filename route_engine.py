import os
import sys

import numpy as np
import traci

repo_path = os.path.dirname(os.path.abspath(__file__))
if repo_path not in sys.path:
    sys.path.append(repo_path)

from enviroment.sumo_env import set_sumo_env
from ray.rllib.policy.policy import Policy


class DRLRouteEngine:
    def __init__(self, checkpoint_path: str):
        print("[DRLRouteEngine] Initializing logical engine...")
        print(f"[DRLRouteEngine] Checkpoint Directory: {checkpoint_path}")

        sumo_cfg_file = os.path.join(repo_path, "sumo", "Esch-Belval.sumocfg")
        sumo_net_file = os.path.join(repo_path, "sumo", "Esch-Belval.net.xml")

        self.sumoCmd, self.sumoNet = set_sumo_env(sumo_cfg_file, sumo_net_file, GUI=False)
        try:
            traci.getConnection("route_engine")
        except Exception:
            traci.start(self.sumoCmd, port=35000, label="route_engine")

        con = traci.getConnection("route_engine")
        self.edges = [edge for edge in con.edge.getIDList() if not edge.startswith(":")]
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edges)}
        con.close()

        print("[DRLRouteEngine] Initializing Ray framework and restoring policy...")
        import ray

        ray.init(ignore_reinit_error=True, log_to_driver=False)
        self.policy = Policy.from_checkpoint(checkpoint_path)
        print("[DRLRouteEngine] Policy loaded.")

    def compute_optimal_route(self, start_edge_id: str, dest_edge_id: str) -> list[str]:
        print(f"[RouteEngine] > Inference Request: Start '{start_edge_id}' => Dest '{dest_edge_id}'")

        if start_edge_id not in self.edge_to_idx:
            print(f"[RouteEngine] ERROR: startEdgeId '{start_edge_id}' is not in the observation space.")
            return []
        if dest_edge_id not in self.edge_to_idx:
            print(f"[RouteEngine] ERROR: destEdgeId '{dest_edge_id}' is not in the observation space.")
            return []

        route = [start_edge_id]
        current_edge = start_edge_id
        visited_edges = {start_edge_id}
        prev_dist = self._distance_to_goal(start_edge_id, dest_edge_id)
        max_dist = max(prev_dist, 1.0)

        step_count = 0
        max_steps = 1000

        while current_edge != dest_edge_id and step_count < max_steps:
            try:
                out_edges = self._outgoing_edges(current_edge)
                if not out_edges:
                    print(f"[RouteEngine] Dead end at edge {current_edge}.")
                    break

                obs = self._build_obs(
                    current_edge=current_edge,
                    dest_edge_id=dest_edge_id,
                    prev_dist=prev_dist,
                    max_dist=max_dist,
                    n_out=len(out_edges),
                    visited_edges=visited_edges,
                    step_count=step_count,
                    max_steps=max_steps,
                )

                action, _, _ = self.policy.compute_single_action(obs)
                if isinstance(action, tuple):
                    action = int(action[0])
                else:
                    action = int(action)

                action_idx = action if action < len(out_edges) else 0
                current_edge = out_edges[action_idx]
                route.append(current_edge)
                visited_edges.add(current_edge)
                prev_dist = self._distance_to_goal(current_edge, dest_edge_id)
                step_count += 1

            except Exception as e:
                print(f"[RouteEngine] Exception during step {step_count}: {e}")
                break

        if current_edge == dest_edge_id:
            print(f"[RouteEngine] Target '{dest_edge_id}' reached.")
        elif step_count >= max_steps:
            print(f"[RouteEngine] Reached max_steps limit ({max_steps}).")

        print(f"[RouteEngine] < Path processed with {step_count} steps.")
        return route

    def _build_obs(
        self,
        current_edge: str,
        dest_edge_id: str,
        prev_dist: float,
        max_dist: float,
        n_out: int,
        visited_edges: set[str],
        step_count: int,
        max_steps: int,
    ):
        dist = self._distance_to_goal(current_edge, dest_edge_id)
        ref = max(max_dist, 1.0)

        obs = np.zeros(len(self.edges) + 6, dtype=np.float32)
        edge_idx = self.edge_to_idx.get(current_edge)
        if edge_idx is not None:
            obs[edge_idx] = 1.0

        obs[-6:] = np.array(
            [
                float(np.clip(dist / ref, 0.0, 1.0)),
                float(np.clip((prev_dist - dist) / ref, -1.0, 1.0)),
                float(min(n_out, 4)) / 4.0,
                0.5,
                1.0 if current_edge in visited_edges else 0.0,
                float(np.clip(step_count / max(max_steps, 1), 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return obs

    def _distance_to_goal(self, current_edge: str, dest_edge_id: str):
        try:
            path, cost = self.sumoNet.getShortestPath(
                self.sumoNet.getEdge(current_edge),
                self.sumoNet.getEdge(dest_edge_id),
            )
            if path is not None and cost is not None:
                return float(max(cost, 0.0))
        except Exception:
            pass
        return 1.0

    def _outgoing_edges(self, current_edge: str):
        try:
            return [edge.getID() for edge in self.sumoNet.getEdge(current_edge).getOutgoing()]
        except Exception:
            return []
