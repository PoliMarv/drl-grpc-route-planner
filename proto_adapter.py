import coSim_pb2

def parse_request_to_edges(request: coSim_pb2.Request) -> tuple[str, str]:
    """
    Extracts logical input (startEdgeId and destEdgeId) from the Protobuf Request.
    Expects values to be correctly placed in the stringArray field of the value parameter.
    """
    if not request.value.HasField("stringArray"):
        raise ValueError("The Request.value field does not contain a stringArray as expected.")
    
    values = request.value.stringArray.values
    if len(values) < 2:
        raise ValueError("The stringArray field does not have enough parameters for startEdgeId and destEdgeId.")
        
    start_edge_id = values[0]
    dest_edge_id = values[-1] # Potentially values[1] depending on how the message is populated
    
    return start_edge_id, dest_edge_id

def create_response_from_route(original_request: coSim_pb2.Request, route: list[str]) -> coSim_pb2.Request:
    """
    Builds a response starting from the original request, replacing 
    the value payload with the resulting list of edges saved in the StringArray.
    """
    response = coSim_pb2.Request()
    response.CopyFrom(original_request)
    
    # Insert the path edges into the response's stringArray
    response.value.stringArray.values[:] = route
    return response
