import os

# Get the directory containing the __init__.py file
init_file_path = os.path.abspath(__file__)
SIM_XML_PATH = os.path.dirname(init_file_path) + "/assets/world.xml"
GREEN_SCREEN_SIM_XML_PATH = os.path.dirname(init_file_path) + "/assets/green_screen_world.xml"