# Import modules
import pickle


def run(x, model_name):
    MODELS_PATH = '../models'

    # Load normalization
    with open(f'{MODELS_PATH}/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Normalize
    x_norm = scaler.transform(x)

    # Load model
    with open(f'{MODELS_PATH}/{model_name}.pkl', 'rb') as file:
        model = pickle.load(file)['model']

    # Predict
    y_hat = model.predict(x_norm)

    return y_hat
