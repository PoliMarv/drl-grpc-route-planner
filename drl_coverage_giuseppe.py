import os
import sys
import logging
import gym
import ray
import traci
import tensorflow as tf
import traci.constants as tc
import sumolib
import numpy as np
import csv
from ray.rllib.algorithms import ppo
# coverage
from coverage_prediction.CoveragePrediction import CoveragePrediction
from enviroment.sumo_env import set_sumo_env
from drl_agent.RewardSystem import Reward



sumoCmd, sumoNet = set_sumo_env("sumo/barisquare.sumo.cfg", "sumo/barisquare.net1.xml", GUI=True)

#print(sumoCmd)



# Edge di partenza e arrivo aggiornati
startEdge = "50317015#0"
revStartEdge= ""
endEdge = "24884043#18"
#endEdge = "24884052#19"
step_index = 0
# 24884043#18
# invece del baseRoute="r_0" che viene dichiarato nei file sumo, lo costruisco qui
"""
traci.start(sumoCmd)
routeID = "my_route"
traci.route.add(routeID, traci.simulation.findRoute(startEdge, endEdge).edges)
edges_in_route = traci.route.getEdges(routeID)

# Stampa gli edge associati alla route
print(f"Edge associati alla route {routeID}: {edges_in_route}")

# Verifica che tutti gli edge esistano nella rete
for edge in edges_in_route:
    if edge in traci.edge.getIDList():
        print(f"Edge {edge} è valido!")
    else:
        print(f"Edge {edge} NON è valido!")
"""

class RoutePlannerDRL(gym.Env):

    ego_idx = -1
    current_ego = "EGO_0"
    optimalRoute = [startEdge]
    prev_dist = 0

    def __init__(self, env_config):
        # DRL inizialization
        traci.start(sumoCmd)
        self.edges = traci.edge.getIDList()
        #print(self.edges)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Discrete(len(self.edges))
        # coverage
        self.coverage = CoveragePrediction()

        # reward
        self.reward = Reward()
        self.old_reward = 0 # for the save_step_reward
        self.old_step = 0 # for the save_step_reward
        self.coverage_list = []

        # first vehicle added to the simulation
        # self.add_vehicle()

    def reset(self, seed=None, options=None):
        # return of the position of the observation space as we return every step
        # the reset in called every time we switch to done true to PPO so we add vehicle here
        self.add_vehicle()
        self.optimalRoute = [startEdge]
        self.reward.reset()

        # sistema questi
        self.save_step_reward(self.coverage_list)
        self.coverage_list = []
        self.old_reward = 0
        # ...
        self.prev_dist = traci.simulation.getDistanceRoad(startEdge, 0, endEdge, 0, False)

        return self.edges.index(startEdge)

    def step(self, action):     
        done = False # i put it here ( before the while) beacuse i use it also to save the rewards
        traci.simulationStep()

        while True:
            ego_values = traci.vehicle.getSubscriptionResults(self.current_ego)

            

            if not ego_values:
                if self.current_ego  in traci.vehicle.getIDList():
                    print(f"Veicolo {self.current_ego} senza valori")
                    traci.simulationStep()
                    ego_values = traci.vehicle.getSubscriptionResults(self.current_ego)
                    print(ego_values)
                # then the vehicle is out of bound
                self.reward.set_out_of_bound()
                done = True

                prev_road = self.optimalRoute[-1]
                reward = self.multi_objective_reward_system(self.coverage_list, prev_road)
                self.old_reward = reward + self.old_reward

                return self.edges.index(prev_road), reward, done, {}
            


            current_road = ego_values[tc.VAR_ROAD_ID]
            # finchè l'auto si trova nel vecchio edge, dobbiamo aspettare
            if self.prev_dist == traci.simulation.getDistanceRoad(current_road, 0, endEdge, 0, False):
                # print("stesse distanze")
                traci.simulationStep()
            else:
                # print(f"strada attuale: {current_road}, strada precendente: {self.optimalRoute[-1]}")
                break

            
        if current_road == endEdge:
            self.reward.set_goal_reached()
            print("🏁 Arrivato a destinazione")
            done = True
        elif current_road == revStartEdge:
            done = True


        # calcoliamo i percorsi disponibili
        outEdges = {}
        try:
            outEdges = sumoNet.getEdge(current_road).getOutgoing()
        except Exception:
            pass
        outEdgesList = []
        for outEdge in outEdges:
            outEdgesList.append(outEdge.getID())


        # applichiamo l'azione
        if len(outEdgesList) > 0:
            # evitiamo d'avere azioni illegali
            new_edge = outEdgesList[action % len(outEdgesList)]

            self.optimalRoute.append(new_edge)
            traci.vehicle.setRoute(self.current_ego, [current_road, new_edge])
            # lets add the coverage of the position to its list
            self.coverage_list.append(self.coverage.predict(traci.vehicle.getPosition(self.current_ego)))
            

        reward = self.multi_objective_reward_system(self.coverage_list, current_road)
        self.old_reward = reward + self.old_reward

        # what was previous now becomes old:
        self.prev_dist = traci.simulation.getDistanceRoad(current_road, 0, endEdge, 0, False)

        return self.edges.index(current_road), reward, done, {}


    def save_step_reward(self, coverage_list):
        csv_file = "reward_cov_g.csv"
        if len(coverage_list) > 0:
            coverage_mean = sum(coverage_list) / len(coverage_list)
        else:
            coverage_mean = 0

        self.old_step = self.old_step + 1
        global step_index
        step_index = self.old_step
        # print(f"Step: {self.old_step}, Reward: {self.old_reward}")
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)            # Scrive i nuovi valori
            writer.writerow([self.old_step, self.old_reward, coverage_mean])
        



    def add_vehicle(self):
        """ Aggiunge un veicolo con un percorso valido, evitando duplicati """
        if (self.ego_idx > -1 and self.current_ego in traci.vehicle.getIDList()):
            #print(f"🔄 Rimuovendo veicolo esistente {self.current_ego}")
            traci.vehicle.unsubscribe(self.current_ego)
            traci.vehicle.remove(self.current_ego)

        self.ego_idx += 1
        self.current_ego = f"EGO_{self.ego_idx}"

        traci.vehicle.add(self.current_ego, routeID="r_0")
        traci.vehicle.subscribe(self.current_ego, (
            tc.VAR_ROUTE_ID,
            tc.VAR_ROAD_ID,
            tc.VAR_POSITION,
            tc.VAR_SPEED,
        ))

        

    def multi_objective_reward_system(self, coverage_list, current_road):
        reward = 0
        # variables of the multiobjective function
        COVERAGE_TRESHOLD = 15
        TARGET_ACQUIRED_WEIGHT = 1000
        ILLEGAL_ACTION_WEIGHT = 500
        DISTANCE_REWARD_WEIGHT = 5 #5 #25
        COVERAGE_REWARD_WEIGHT = 15 #15#15
        MIN_COVERAGE_REWARD_WEIGHT = 1000

        # 1: calculate the reward for targt acquired or illegal action
        # it describes if the car arrived at the target, as int insttead of bool for the multiobjective fun
        target_acquired = 0
        # it describes if the car executes an illegal action
        illegal_action = 0
        if self.reward.goal_reached == 1:
            target_acquired = 1
        elif self.reward.out_of_bound == 1:
            illegal_action = -1
        

        # 2: calculate the reward for the distance
        current_dist = traci.simulation.getDistanceRoad(current_road, 0, endEdge, 0, False)
        distance_reward = 0
        if current_dist < self.prev_dist:
            distance_reward = 1
        else:
            distance_reward = -1
        
        
        # 3: calculate the reward for increasing coverage
        coverage_reward = 0
        if len(coverage_list) >= 2:
            current_coverage = coverage_list[-1]
            previous_coverage = coverage_list[-2]
            if current_coverage > previous_coverage:
                coverage_reward = 1
            else:
                coverage_reward = -1

        # 4: calculate the reward for the minimum coverage
        min_coverage_reward = 0 # only if under the threshold there is a punischment
        if len(coverage_list) > 0:
            if coverage_list[-1] < COVERAGE_TRESHOLD:
                min_coverage_reward = -1

        
        # MULTI OBJECTIVE FUNCTION
        # the variable used are: target_acquired, illegal_action, distance_reward, coverage_reward, min_coverage_reward
        #print(current_dist)
        
        reward = (TARGET_ACQUIRED_WEIGHT * target_acquired) + (ILLEGAL_ACTION_WEIGHT * illegal_action) + (DISTANCE_REWARD_WEIGHT * distance_reward) + (COVERAGE_REWARD_WEIGHT * coverage_reward) + (MIN_COVERAGE_REWARD_WEIGHT * min_coverage_reward)
        # print(reward)
        print(f"reward: {reward} ...target_acquired: {target_acquired}, illegal_action: {illegal_action}, distance_reward: {distance_reward}, coverage_reward: {coverage_reward}, min_coverage_reward: {min_coverage_reward}")
        return reward


# ✅ Inizializza Ray
ray.init()

algo = ppo.PPO(env=RoutePlannerDRL, config={
    "env_config": {},
    "num_workers": 0
})


# salvo la ricompensa in un csv
# se il csv esiste lo pulisco
csv_file_path = "reward_cov_g.csv"
if os.path.exists(csv_file_path):
    with open(csv_file_path, 'w') as f:
        pass
    

# Loop di addestramento con gestione errori
while step_index < 400000:
    try:
        res = algo.train()
        if "episode_reward_mean" in res:
            print(f"🏆 Ricompensa media per episodio: {res['episode_reward_mean']}")
        else:
            print("⚠️ Nessuna ricompensa registrata, dettagli:", res)
        # checkpoint 
        print(step_index)
        #if res["episode_reward_mean"] > 3000:

        #    break
    except KeyError as e:
        print(f"❌ Errore: {e} - Struttura della risposta: {res}")

import sys
sys.setrecursionlimit(5000) 
# da problemi perchè proviamo a salvare anche altre robe come il coverage prediction e diventa ricorsivo ..
# # checkpoint = algo.save_checkpoint("my_checkpoints")
# print(f"📦 Salvataggio checkpoint: {checkpoint}")
policy = algo.get_policy()
policy.export_checkpoint("my_checkpoints_giuseppe_cov")
# policy.save("my_policy")
#tf.saved_model.save(policy.model, "ppo_model_tf")
# torch.save(policy.model.state_dict(), "ppo_weights.pt")