# Rabbit Object Tracker

This project is a Python project using PyTorch to train and test a model for object tracking.

## Project Structure

The project has the following files and directories:

- `data/raw`: This folder is used to store the raw data for training and testing the model.

- `data/processed`: This folder is used to store the processed data after pre-processing.

- `models/model.py`: This file contains the Python code for the model. It exports a class `Model` which implements the PyTorch model architecture.

- `notebooks/exploration.ipynb`: This Jupyter notebook is used for data exploration and analysis.

- `src/train.py`: This file contains the Python code for training the model. It imports the `Model` class from `models/model.py` and uses it to train the model on the processed data.

- `src/test.py`: This file contains the Python code for testing the trained model. It imports the `Model` class from `models/model.py` and uses it to evaluate the model's performance on test data.

- `src/utils.py`: This file contains utility functions that are used in the training and testing process.

- `requirements.txt`: This file lists the dependencies required for the project. It specifies the PyTorch library as a dependency.

## Usage

To use this project, follow these steps:

1. Clone the repository.

2. Install the required dependencies by running the following command:

   ```
   pip install -r requirements.txt
   ```

3. Preprocess the raw data and store it in the `data/processed` directory.

4. Train the model by running the following command:

   ```
   python src/train.py
   ```

5. Test the trained model by running the following command:

   ```
   python src/test.py
   ```

For more details on the project and its implementation, refer to the source code and the `notebooks/exploration.ipynb` notebook.

```

This file provides an overview of the project structure and instructions on how to use it.