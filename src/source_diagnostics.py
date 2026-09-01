import argparse
from cs_dfm.config import load_config
from cs_dfm.visualization import source_diagnostics

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--lambdas", type=float, nargs="+", default=[0,.1,.2,.4,.6,.8,1])
    p.add_argument("--temperatures", type=float, nargs="+", default=[.5,1,1.5,2,4]); p.add_argument("--max-samples", type=int); p.add_argument("--output", default="visualizations/source_diagnostics"); args=p.parse_args()
    source_diagnostics(load_config(args.config), args.lambdas, args.temperatures, args.output, args.max_samples)

