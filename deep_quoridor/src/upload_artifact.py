#!/usr/bin/env python3
import argparse
import os
import re

import wandb


def upload_artifact(file_path):
    """
    Upload a file to wandb as a new version of an existing model artifact.
    The artifact name is determined from the file name.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return False
    # Extract artifact name from file path
    # Assuming naming convention like: model_dqn_v1.zip or ppo_quoridor_v2.zip
    # The v1, v2, etc. is added by the TrainableAgent class
    file_name = os.path.basename(file_path)
    match = re.search(r"([a-zA-Z0-9_]+)_C[a-zA-Z0-9_]+_(?:\d+)?\.(?:zip|pkl|h5|pt|pth)", file_name)

    if not match:
        print(f"Error: Could not determine artifact name from {file_name}")
        print("Expected format: model_name[_v1].extension")
        return False

    artifact_name = match.group(1)

    # Initialize wandb
    run = wandb.init(project="deep-quoridor", job_type="model-upload")

    # Create a new artifact
    artifact = wandb.Artifact(name=artifact_name, type="model", description=f"New version of {artifact_name} model")

    # Add file to artifact
    artifact.add_file(file_path)

    # Log the artifact
    run.log_artifact(artifact)
    print(f"Successfully uploaded {file_path} as artifact '{artifact_name}'")

    wandb.finish()
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload a file to W&B as a model artifact")
    parser.add_argument("file", help="Path to the file to upload")
    args = parser.parse_args()

    upload_artifact(args.file)


if __name__ == "__main__":
    main()
