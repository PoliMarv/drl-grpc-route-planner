import os, sys
import sumolib
from pathlib import Path

def set_sumo_env(sumo_cfg_file_path: str, sumo_net_file_path :str, GUI :bool):
    """
    Set the SUMO environment with the configuration file and the network file
    :param sumo_cfg_file_path: absolute path to the SUMO configuration file 
    :param sumo_net_file_path: absolute path to the SUMO network file
    :param GUI: boolean to set the SUMO GUI or not
    :return: sumoCmd, sumoNet, commands to run SUMO and get infos of the road network
    """

    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # SUMO configuration with or without GUI (cross-platform)
    is_windows = os.name == "nt"
    if GUI:
        binary_name = "sumo-gui.exe" if is_windows else "sumo-gui"
    else:
        binary_name = "sumo.exe" if is_windows else "sumo"

    sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', binary_name)

    # takes the command and the net from the folder called sumo
    sumoCmd = [sumoBinary, "-c", str(Path(sumo_cfg_file_path))]
    sumoNet = sumolib.net.readNet(str(Path(sumo_net_file_path)))

    return sumoCmd, sumoNet

# TEST
"""
print("Test set_sumo_env")
sumo_cfg_file_path = "sumo/Esch-Belval.sumocfg"
sumo_net_file_path = "sumo/Esch-Belval.net.xml"
print(sumo_cfg_file_path)
GUI = True
sumoCmd, sumoNet = set_sumo_env(sumo_cfg_file_path, sumo_net_file_path, GUI)
print(sumoCmd)
"""
