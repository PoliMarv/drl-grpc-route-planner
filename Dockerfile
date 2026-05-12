FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SUMO_HOME=/usr/share/sumo

WORKDIR /app

# Install SUMO runtime (headless) and basic build/runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    sumo \
    sumo-tools \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

# Copy only inference/gRPC runtime assets.
COPY grpc_server.py grpc_service.py route_engine.py proto_adapter.py ./
COPY coSim.proto coSim_pb2.py coSim_pb2_grpc.py ./
COPY test_client.py ./
COPY env ./env
COPY enviroment ./enviroment
COPY coverage_prediction/CoveragePrediction.py ./coverage_prediction/CoveragePrediction.py
COPY drl_agent/RewardSystem.py ./drl_agent/RewardSystem.py
COPY sumo ./sumo
COPY results_full_objective/checkpoint_best ./results_full_objective/checkpoint_best

EXPOSE 50051

# Default: serve best full-objective policy on all interfaces.
CMD ["python", "grpc_server.py", "--checkpoint", "full_best", "--port", "50051", "--bind", "0.0.0.0"]
