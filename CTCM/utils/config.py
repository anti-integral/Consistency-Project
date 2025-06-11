import yaml, argparse

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_cifar.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["resume"] = args.resume
    return cfg
