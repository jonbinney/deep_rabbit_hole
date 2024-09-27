## Logging Experiments with MLFlow

### Starting the mlflow tracking server

In a terminal that has the virtual environment active, start the mlflow server.

```
mlflow server --host 127.0.0.1 --port 8080
```

Alternatively, you can run the "Start local MLflow server" task in vscode by
pressing `Ctrl+Shift+P`, searching for "Run Task", then search for "Start local
MLflow server". This will create a new terminal inside of VSCode running the
local server.

Note: By default, the server creates `mlartifacts/` and `mlruns/` in the current
directory for storing data about runs.

### Logging a run and viewing the results

Now run the inference script. In the console output of the script, MLFlow will
print something like "View experiment at <URL>". If you open that URL in a
browser window, you should see a list of the "runs" of this "experiment". Click
on the most recent one to see the results of this particular run.

There are a few tabs near the top of the page for the run. On the "Overview" tab
you can see the parameters used for this run. On the "System metrics" tab you
can see the GPU memory, CPU, etc. used at various points during the run. On the
"Artifacts" tab you can see the files created during the run (for example the
annotations created during an inference run.)

### Disabling mlflow logging

Sometimes you may want to run your code without the mlflow server running. To do
that, unset the MLFLOW_TRACKING_URI environment parameter. This actually sets
the tracking uri to `file:///tmp/mlruns` so that logging happens to a temporary
location on disk instead of the mlflow tracking server.
