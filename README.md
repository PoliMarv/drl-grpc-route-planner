# DRL gRPC Route Planner (SUMO + PPO)

This repository contains a Deep Reinforcement Learning route-planning system based on:
- SUMO traffic simulation
- Ray RLlib PPO policy
- gRPC inference service for external clients/co-simulators

The service receives two edge IDs (`start_edge`, `dest_edge`) and returns the full edge sequence of the computed route.

## Contents
1. Project Goals
2. Repository Structure
3. Runtime Architecture
4. Prerequisites
5. Local Setup
6. Training
7. gRPC API
8. Docker Usage
9. Testing
10. Troubleshooting
11. Publishing
12. Model Weights Strategy

---

## 1) Project Goals

The DRL agent is trained to optimize routing with a multi-objective reward:
- reach destination
- improve distance progression
- improve SINR/coverage
- improve QoS

The trained policy is exposed through gRPC for production-style inference.

---

## 2) Repository Structure

Main runtime files:
- `grpc_server.py`: gRPC server entrypoint
- `grpc_service.py`: protobuf service implementation
- `route_engine.py`: route inference engine + checkpoint loading
- `proto_adapter.py`: protobuf <-> Python conversion helpers
- `coSim.proto`, `coSim_pb2.py`, `coSim_pb2_grpc.py`: protocol and generated stubs
- `test_client.py`: local gRPC test client

Training/environment files:
- `train_full_objective.py`
- `train_distance_only.py`
- `env/route_planner_env.py`
- `enviroment/sumo_env.py`
- `coverage_prediction/CoveragePrediction.py`
- `drl_agent/RewardSystem.py`

Map files:
- `sumo/Esch-Belval.*`

---

## 3) Runtime Architecture

### Training phase
1. PPO interacts with SUMO via TraCI.
2. Observations and reward are computed from route state and network context.
3. Policy checkpoints are exported.

### Inference phase
1. Client calls `coSim.CoSim/GetAttribute`.
2. Service extracts `[start_edge, dest_edge]` from `Request.value.stringArray.values`.
3. `DRLRouteEngine` loads policy checkpoint and iteratively predicts next edges.
4. Full route is returned in `Response.value.stringArray.values`.

---

## 4) Prerequisites

- Python 3.10 recommended
- SUMO installed on host, with `SUMO_HOME` configured
- Docker (optional, recommended for distribution)

Dependency files:
- `requirements.txt`: full local stack (training + inference + plotting)
- `requirements-docker.txt`: Docker-focused runtime dependencies

---

## 5) Local Setup

Example (Conda):

```bash
conda create -n drl_new_env python=3.10
conda activate drl_new_env
pip install -r requirements.txt
```

Verify `SUMO_HOME` is available in your environment before running training/inference.

---

## 6) Training

Full objective training:

```bash
python train_full_objective.py --steps 800000 --gpu
```

Distance-only training:

```bash
python train_distance_only.py --steps 500000 --gpu
```

Common options (script-dependent):
- `--gpu`
- `--gui`
- `--resume <checkpoint_path>`

---

## 7) gRPC API

Service/method:
- `coSim.CoSim/GetAttribute`

Required request fields:
- `attribute = ROUTE`
- `type = GET`
- `id = <request id>`
- `value.stringArray.values = [start_edge, dest_edge]`

Response payload:
- `value.stringArray.values = [edge_1, edge_2, ..., edge_n]`

Error behavior:
- `NOT_FOUND`: invalid edge IDs or no route found
- `INTERNAL`: engine/checkpoint/runtime failure

Start server locally:

```bash
python grpc_server.py --checkpoint full_best --port 50051 --bind 127.0.0.1
```

Checkpoint aliases:
- `full_best` -> `results_full_objective/checkpoint_best`
- `full_final` -> `results_full_objective/checkpoint`

---

## 8) Docker Usage

Build image:

```bash
docker build -t drl-grpc-route:full-best .
```

Run container:

```bash
docker run --rm --name drl_route_test --shm-size=2g -p 50052:50051 drl-grpc-route:full-best
```

Notes:
- Host `50052` forwards to container `50051`.
- Clients must call `127.0.0.1:50052`.
- `--shm-size=2g` is recommended for Ray shared-memory stability.

---

## 9) Testing

### A) Python client test

```bash
python test_client.py --host 127.0.0.1 --port 50052 --start-edge 361567664 --dest-edge 291922340#2
```

Expected result:
- response received
- non-empty route list

### B) grpcurl (via Docker, no local install required)

```bash
docker run --rm -v "${PWD}:/protos" fullstorydev/grpcurl \
  -plaintext \
  -import-path /protos \
  -proto coSim.proto \
  -d '{"attribute":"ROUTE","type":"GET","id":"VEHICLE_TEST_01","value":{"stringArray":{"values":["361567664","291922340#2"]}}}' \
  host.docker.internal:50052 coSim.CoSim/GetAttribute
```

Known valid test pair:
- start edge: `361567664`
- destination edge: `291922340#2`

---

## 10) Troubleshooting

### Port already in use
Use a different host port:

```bash
docker run --rm --shm-size=2g -p 50053:50051 drl-grpc-route:full-best
```

Then point clients to `50053`.

### `grpcurl` not found
Use `test_client.py` or grpcurl via Docker.

### `Unable to start the Route Engine`
Check:
1. checkpoint alias/path
2. SUMO files availability
3. container logs (`docker logs <container_name>`)

### First call is slower
Normal behavior: first request triggers lazy policy/engine initialization.

---

## 11) Model Weights Strategy

For GitHub, storing large model weights directly in the repository is usually a bad idea.

If you need the weights, write me an email.
