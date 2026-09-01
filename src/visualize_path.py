import argparse
from cs_dfm.visualization import visualize_paths

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output", default="visualizations/paths"); p.add_argument("--num-classes", type=int, default=20); args=p.parse_args()
    configs=[
      {"name":"linear","type":"two_term","scheduler":"linear"},
      *[{"name":f"power-{x:g}","type":"two_term","scheduler":"power","power":x} for x in [.5,1,2,4]],
      {"name":"three-term","type":"three_term","scheduler":"power_uniform_bump","power":2,"uniform_strength":.3},
    ]
    visualize_paths(configs,args.num_classes,args.output)

