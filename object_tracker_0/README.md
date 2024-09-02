# Rabbit Object Tracker

This project is a Python project using PyTorch to train and test a model for object tracking.

# Running inference

Use the Run configuration "Inference".
I used it with the rabbits_2024_08_12_25_15sec video and it successfully detects the rabbit
on many frames (it only runs detection every 10 frames or so) and writes the result in an
annotation file. If you take that annotation file and uploaded on CVAT, you can see it matches
the video appropriately.

# Installing Grounding DINO

These instructions need to be done only once to set-up your environment so that you can run
inference using Grounding DINO

## Download Grounding DINO code

It should be installed as an external subtree under external/GroundingDINO.
If it doesn't show-up in your repo, it's done like so:

   git remote add GroundingDINO https://github.com/IDEA-Research/GroundingDINO.git
   git subtree add --prefix=external/GroundingDINO GroundingDINO main --squash

## Install CUDA dependencies

You can use `nvidia-smi` to check your current CUDA version. Mine (Julian) was 12.5
at the time of writing this, thus the versions chosen in the next step.

I had to install the following in Ubuntu 20.04 for the subsequent pip installation
to work:

   sudo apt install libcusparse-dev-12-5 libcusolver-dev-12-5 libcublas-dev-12-5

I also set-up CUDA_HOME just in case

   export CUDA_HOME=/usr/local/cuda

Which in my case it was a symlink to a symlink to a symlink to /usr/local/cuda-12.5

## pip install

The instructions are [in the Grounding DINO repo](https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#hammer_and_wrench-install)
but basically it comes down to:

 - Activate your venv (VSCode does it for you I think)
 - Go to external/GroundingDINO
 - pip install -e .