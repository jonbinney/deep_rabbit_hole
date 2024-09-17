## Running on a Cloud GPU

This guide assumes that you have already started a cloud machine using some compute provider,
e.g. Lambda Labs. It assumes that the remote machine comes with a relatively recent version of
python, pip, and virtualenv, and has CUDA libraries installed already. All commands below are
assumed to be run from the root directory of this repo.

To make things easier, store the IP of the cloud machine in an environment variable.

```
export CLOUD_IP=<ip of cloud machine>
```

Transfer our code to the remote machine, leaving out directories we don't need.

```
rsync -av --exclude .venv --exclude .git $PWD ubuntu@$CLOUD_IP:
```

On the remote machine, create a virtual environment, activate it,  and install the our dependencies.
This should take a couple minutes.

```
cd ~/deep_rabbit_hole
virtualenv .venv
. .venv/bin/activate
pip install -r object_tracker_0/requirements.txt
```

Install segment-anything-2 (into the virtual environment). This will also take a couple of minutes.

```
pip install -e external/segment-anything-2/
```

Now you can run our scripts. For example to run inference:

```
python object_tracker_0/src/inference.py -v datasets/rabbits_2024_08_12_25_15sec/video/rabbits_2024_08_12_15sec.mp4 -a datasets/rabbits_2024_08_12_25_15sec/annotations/test_v1.json -w /tmp/15sec
```
