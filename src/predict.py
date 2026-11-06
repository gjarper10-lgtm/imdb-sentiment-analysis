"""Predict a new review using a saved model bundle, without training or downloads."""
import argparse
import json
import torch
from .artifacts import load_bundle, inference_loader
from .engine import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    model, spec, info = load_bundle(args.model_dir, device)
    probabilities, _, _ = predict(model, inference_loader([args.text], spec, info), device, amp=False)
    probability = float(probabilities[0])
    print(json.dumps({'sentiment': 'positive' if probability >= 0.5 else 'negative',
                      'probability_positive': probability}, indent=2))


if __name__ == '__main__':
    main()
