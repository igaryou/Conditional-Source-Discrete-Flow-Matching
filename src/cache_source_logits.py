import argparse
from cs_dfm.cache import create_source_cache
from cs_dfm.config import load_config

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--splits", nargs="+", default=["train", "val"]); args = p.parse_args()
    create_source_cache(load_config(args.config), args.splits)

