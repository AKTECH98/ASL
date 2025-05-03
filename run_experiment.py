import yaml
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config['model']['type']
    note = config['output']['note']
    hidden_size = config['model']['hidden_size']
    dropout = config['model'].get('dropout', 0)
    bidir = config['model'].get('bidirectional', False)
    lr = config['train']['learning_rate']
    epochs = config['train']['epochs']
    exp_name = f"{model_name}__hs-{hidden_size}__bi-{str(bidir)[0]}__do-{dropout}__lr-{lr}__ep-{epochs}__{note}"

    # Train
    subprocess.run([
        "python", "train.py",
        "--config", args.config, "--exp_name", exp_name,
    ])

    # # Evaluate
    subprocess.run([
        "python", "evaluate.py",
        "--config", args.config,
        "--exp_name", exp_name
    ])

if __name__ == "__main__":
    main()
