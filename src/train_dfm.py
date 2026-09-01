import argparse
from cs_dfm.config import load_config
from cs_dfm.train import train_dfm

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); args = p.parse_args()
    train_dfm(load_config(args.config))

