import torch
from models.model import Model
from src.utils import load_data, preprocess_data

def test_model():
    # Load test data
    test_data = load_data('data/processed/test_data.csv')

    # Preprocess test data
    preprocessed_test_data = preprocess_data(test_data)

    # Load trained model
    model = Model()
    model.load_state_dict(torch.load('models/trained_model.pth'))
    model.eval()

    # Evaluate model on test data
    with torch.no_grad():
        predictions = model(preprocessed_test_data)
    
    # Perform further analysis on predictions
    # ...

if __name__ == '__main__':
    test_model()