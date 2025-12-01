import yaml
import os

# Get the path to the config directory
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

# Load the YAML config file
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
