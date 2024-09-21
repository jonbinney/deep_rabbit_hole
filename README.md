## Running on a Cloud GPU

### Setting up the remote machine

This guide assumes that you have already started a cloud machine using some compute provider,
e.g. Lambda Labs. It assumes that the remote machine comes with a relatively recent version of
python, pip, and virtualenv, and has CUDA libraries installed already. All commands below are
assumed to be run from the root directory of this repo.

The `remote_setup` script copies our repo to the remote machine and installs dependencies in a
virtual environment.

```
./scripts/remote_setup <remote_user> <remote_address>
```

Now you can run our scripts. For example to run inference, ssh to the remote machine and run:

```
cd deep_rabbit_hole
. .venv/bin/activate
python object_tracker_0/src/inference.py -v datasets/rabbits_2024_08_12_25_15sec/video/rabbits_2024_08_12_15sec.mp4 -a datasets/rabbits_2024_08_12_25_15sec/annotations/test_v1.json -w /tmp/15sec
```

### Using VSCode to edit and run code remotely

In VSCode, click the `><` icon in the bottom left and then click the "SSH" option in the menu that
appears. Enter the `ubuntu@<ip_address>` for the machine you created on the lambdalabs site. Once
connected, choose "Open Folder" in the VSCode "File" menu, and open the "deep_rabbit_hole" folder.

Some useful extensions including "Python" and "debugpy" are automatically installed on by vscode on the server
which runs on the remote host. This may take a minute to finish installing before you can use them. Once they
are ready, you should be able to run and debug our scripts just like you would locally.
