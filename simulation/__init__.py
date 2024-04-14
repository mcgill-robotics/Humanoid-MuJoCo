import os
from gymnasium.envs.registration import register

# Get the directory containing the __init__.py file
init_file_path = os.path.abspath(__file__)
SIM_XML_PATH = os.path.dirname(init_file_path) + "/assets/world.xml"
GREEN_SCREEN_SIM_XML_PATH = (
    os.path.dirname(init_file_path) + "/assets/green_screen_world.xml"
)

# Register environments
register(
    id="GPUHumanoid",
    order_enforce=False,
    entry_point="simulation.gpu_vec_env:GPUVecEnv",
    vector_entry_point="simulation.gpu_vec_env:GPUVecEnv",
    disable_env_checker=True,
)
register(id="CPUHumanoid", entry_point="simulation.cpu_env:CPUEnv")
