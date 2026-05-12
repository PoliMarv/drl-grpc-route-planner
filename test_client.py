import argparse

import grpc

import coSim_pb2
import coSim_pb2_grpc


def build_request(vehicle_id: str, start_edge: str, dest_edge: str) -> coSim_pb2.Request:
    request = coSim_pb2.Request()
    request.attribute = coSim_pb2.Request.ROUTE
    request.type = coSim_pb2.Request.GET
    request.id = vehicle_id
    request.value.stringArray.values.extend([start_edge, dest_edge])
    return request


def run_test(host: str, port: int, start_edge: str, dest_edge: str, vehicle_id: str):
    target = f"{host}:{port}"
    print(f"Attempting to connect to the gRPC Server on {target}...")

    with grpc.insecure_channel(target) as channel:
        stub = coSim_pb2_grpc.CoSimStub(channel)

        request = build_request(vehicle_id, start_edge, dest_edge)
        print(f"Sending ROUTE request. Start: '{start_edge}', Dest: '{dest_edge}'")

        try:
            response = stub.GetAttribute(request)
            received_path = list(response.value.stringArray.values)

            print("=" * 54)
            print("RESPONSE RECEIVED FROM THE DRL SERVICE")
            print(f"Path length: {len(received_path)}")
            print(f"Complete path: {received_path}")
            print("=" * 54)
        except grpc.RpcError as err:
            print(f"gRPC communication error: {err.details()}")


def main():
    parser = argparse.ArgumentParser(description="Simple gRPC client test for ROUTE requests")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--start-edge", type=str, default="361567664")
    parser.add_argument("--dest-edge", type=str, default="291922340#2")
    parser.add_argument("--vehicle-id", type=str, default="VEHICLE_TEST_01")
    args = parser.parse_args()

    run_test(
        host=args.host,
        port=args.port,
        start_edge=args.start_edge,
        dest_edge=args.dest_edge,
        vehicle_id=args.vehicle_id,
    )


if __name__ == "__main__":
    main()