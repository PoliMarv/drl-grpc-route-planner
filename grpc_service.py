import grpc
import coSim_pb2
import coSim_pb2_grpc
from google.protobuf.empty_pb2 import Empty

from route_engine import DRLRouteEngine
from proto_adapter import parse_request_to_edges, create_response_from_route

class CoSimService(coSim_pb2_grpc.CoSimServicer):
    def __init__(self, checkpoint_path: str):
        # Prepare the Engine, deferring startup to the RPC 'Start' directive
        # (or lazy fallback in 'GetAttribute')
        self.route_engine = None
        self.checkpoint_path = checkpoint_path

    def Start(self, request, context):
        """Initializes Server resources by recreating the PPO environment and TraCI in memory."""
        print("[CoSimService] Received RPC Start. Initializing Neural Network...")
        try:
            if not self.route_engine:
                self.route_engine = DRLRouteEngine(self.checkpoint_path)
            else:
                print("[CoSimService] Neural Network already active in memory.")
        except Exception as e:
            print(f"[CoSimService] Critical error loading Neural Network: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            
        return Empty()

    def Finish(self, request, context):
        """Cleans up resources before closing."""
        print("[CoSimService] Received RPC Finish. Clearing resources...")
        if self.route_engine:
            self.route_engine = None
        return Empty()

    def GetAttribute(self, request, context):
        """
        Queries the Neural Network to resolve State into Action.
        """
        attr_name = coSim_pb2.Request.Attribute.Name(request.attribute)
        print(f"[CoSimService] Received RPC GetAttribute, attribute: {attr_name}")
        
        # ROUTE = 1 according to coSim.proto (Enum Attribute)
        if request.attribute == coSim_pb2.Request.ROUTE:
            # Lazy fallback: if the client did not send Start(), initialize on the fly
            if not self.route_engine:
                print("[CoSimService] ⚠️ WARNING: Engine not initialized by Start, performing Real-time loading...")
                try:
                    self.route_engine = DRLRouteEngine(self.checkpoint_path)
                except Exception as e:
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details("Unable to start the Route Engine. Check the Checkpoint path.")
                    return coSim_pb2.Request()
                    
            try:
                # 1. Protocol Translation -> Local Parameters
                start_edge, dest_edge = parse_request_to_edges(request)
                
                # 2. Get Weight Prediction
                route = self.route_engine.compute_optimal_route(start_edge, dest_edge)
                
                if len(route) == 0:
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details("Path is neural network unexplorable or Edges do not exist")
                    return coSim_pb2.Request()
                    
                # 3. Repack Local Parameters -> Protocol and Dispatch
                response = create_response_from_route(request, route)
                return response
            
            except Exception as e:
                print(f"[CoSimService] Severe error in gRPC processing: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
                return coSim_pb2.Request()
                
        # Default fallback
        return request

    # Empty Stub implementations for other methods required by the proto
    def ExecuteOneTimeStep(self, request, context):
        return coSim_pb2.StepReply(status="OK")

    def GetManagedHosts(self, request, context):
        return coSim_pb2.VehicleList()

    def InsertHost(self, request, context):
        return Empty()

    def DeleteHost(self, request, context):
        return Empty()

    def SetAttribute(self, request, context):
        return Empty()

    def QueryRequest(self, request, context):
        return coSim_pb2.Requests()
