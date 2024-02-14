import os

# Get the directory containing the __init__.py file
init_file_path = os.path.abspath(__file__)
SIM_XML_PATH = os.path.dirname(init_file_path) + "/simulation/assets/world.xml"