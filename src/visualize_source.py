import argparse
from cs_dfm.config import load_config
from cs_dfm.visualization import visualize_source

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--index", type=int, default=0)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0,.1,.2,.4,.6,.8,1]); p.add_argument("--temperatures", type=float, nargs="+", default=[.5,1,1.5,2,4])
    p.add_argument("--seeds", type=int, nargs="+", default=[42,43,44,45]); p.add_argument("--output", default="visualizations/source_sweep"); args = p.parse_args()
    print(visualize_source(load_config(args.config), args.index, args.lambdas, args.temperatures, args.seeds, args.output))

