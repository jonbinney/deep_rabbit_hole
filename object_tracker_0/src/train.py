import torch
from models.model import Model
from src.utils import preprocess_data, train_model

# Load and preprocess the data
data = preprocess_data('data/processed')

# Initialize the model
model = Model()

# Train the model
train_model(model, data)

# Save the trained model
torch.save(model.state_dict(), 'models/trained_model.pth')