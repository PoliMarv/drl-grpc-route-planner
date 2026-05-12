import argparse
import os
import time
from concurrent import futures

import grpc

import coSim_pb2_grpc
from grpc_service import CoSimService


def resolve_checkpoint_path(checkpoint_arg: str) -> str:
    """Resolve checkpoint path and support project aliases."""
    aliases = {
        "full_best": os.path.join("results_full_objective", "checkpoint_best"),
        "full_final": os.path.join("results_full_objective", "checkpoint"),
    }
    checkpoint_value = aliases.get(checkpoint_arg, checkpoint_arg)
    return os.path.abspath(checkpoint_value)


def bind_server(server: grpc.Server, port: int, bind_host: str) -> str:
    """
    Bind with fallback support when IPv6 is not available on the host.
    Returns the address that was successfully bound.
    """
    if bind_host == "auto":
        candidates = [f"[::]:{port}", f"0.0.0.0:{port}", f"127.0.0.1:{port}"]
    else:
        candidates = [f"{bind_host}:{port}"]

    last_error = None
    for address in candidates:
        try:
            bound_port = server.add_insecure_port(address)
            if bound_port:
                return address
        except RuntimeError as err:
            last_error = err
            continue

    if last_error:
        raise last_error
    raise RuntimeError(f"Unable to bind gRPC server on any candidate address: {candidates}")


def serve():
    parser = argparse.ArgumentParser(
        description="Start the gRPC Server interfacing with the DRL Agent."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="full_best",
        help=(
            "Relative/absolute path of the policy checkpoint folder, "
            "or alias: full_best | full_final"
        ),
    )
    parser.add_argument("--port", type=int, default=50051, help="Allocated TCP port for gRPC")
    parser.add_argument(
        "--bind",
        type=str,
        default="auto",
        help="Bind host/address. Use 'auto' for IPv6->IPv4 fallback, or set e.g. 0.0.0.0 / 127.0.0.1",
    )
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(
            f"WARNING: The specified checkpoint folder '{checkpoint_path}' does not physically exist on disk."
        )
        print("If DRL calls fail, this is the root cause.")
    elif not os.path.exists(os.path.join(checkpoint_path, "policy_state.pkl")):
        print(
            f"WARNING: '{checkpoint_path}' exists but does not contain policy_state.pkl. "
            "Use a policy checkpoint exported with export_checkpoint."
        )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    coSim_pb2_grpc.add_CoSimServicer_to_server(
        CoSimService(checkpoint_path=checkpoint_path), server
    )

    bind_port = bind_server(server, args.port, args.bind)
    server.start()

    print(f"Smart gRPC Server active and listening on: {bind_port}")
    print(f"DRL checkpoint input: {args.checkpoint}")
    print(f"Resolved checkpoint path: {checkpoint_path}")
    print("Press Ctrl+C to stop listening.")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("\nHalting the gRPC server...")
        server.stop(0)


if __name__ == "__main__":
    serve()
