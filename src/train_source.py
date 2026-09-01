import argparse
from cs_dfm.config import load_config
from cs_dfm.train import train_source

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); args = p.parse_args()
    train_source(load_config(args.config))

