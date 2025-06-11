"""
Simple wrapper; relies on `pip install torch-fidelity`.
"""
import argparse, os, torch
from torch_fidelity import calculate_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--real_dir", required=True)
    args = parser.parse_args()

    metrics = calculate_metrics(
        input1=args.gen_dir,
        input2=args.real_dir,
        fid=True, isc=False, kid=False, verbose=False,
        cuda=torch.cuda.is_available()
    )
    print("FID:", metrics["frechet_inception_distance"])

if __name__ == "__main__":
    main()
