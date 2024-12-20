import yaml
import os

# configディレクトリのパスを取得
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

# YAMLファイルを読み込む
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
