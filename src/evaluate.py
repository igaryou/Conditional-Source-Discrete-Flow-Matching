import argparse, json
from cs_dfm.config import load_config
from cs_dfm.evaluate import evaluate_dfm

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config"); p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",default="outputs/evaluation"); p.add_argument("--fixed-t",type=float)
    p.add_argument("--generative-steps",type=int); a=p.parse_args()
    print(json.dumps(evaluate_dfm(load_config(a.config) if a.config else None,a.checkpoint,a.output,a.fixed_t,a.generative_steps),indent=2))
