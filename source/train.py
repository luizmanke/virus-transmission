# Import modules
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

# Models
MODELS = [
    {
        'name': 'mlp',
        'model': MLPRegressor(random_state=1),
        'params': {
            'hidden_layer_sizes': (10, 100),
            'alpha': (1e-6, 1e+6, 'log-uniform')
        }
    },
    {
        'name': 'elastic_net',
        'model': ElasticNet(random_state=1),
        'params': {
            'alpha': (1e-6, 1e+6, 'log-uniform'),
            'l1_ratio': (0, 1, 'log-uniform')
        }
    },
    {
        'name': 'knn',
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': (5, 50)
        }
    }
]


def run(x, y):
    MODELS_PATH = '../models'

    # Normalize
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x)

    # Save normalization
    with open(f'{MODELS_PATH}/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    # Hyperparameter tunning
    for item in MODELS:

        search = BayesSearchCV(
            item['model'],
            item['params'],
            scoring=make_scorer(business_function, greater_is_better=False),
            cv=3,
            n_iter=20,  # only 20 iterations in order to speed up the process
            return_train_score=True,
            random_state=1
        )
        search.fit(x_norm, y)

        cv_results = search.cv_results_
        best_index = search.best_index_
        results = {
            'train_mean': -1 * cv_results['mean_train_score'][best_index],
            'train_std': cv_results['std_train_score'][best_index],
            'validation_mean': -1 * cv_results['mean_test_score'][best_index],
            'validation_std': cv_results['std_test_score'][best_index],
            'model': search.best_estimator_
        }

        # Save model
        with open(f'{MODELS_PATH}/{item["name"]}.pkl', 'wb') as file:
            pickle.dump(results, file)


def business_function(y, y_hat):
    '''
    This business function generates larger errors when the estimation
    rate is lower than the actual rate to prevent false negatives.

    False negative (high error):
        estimation rate = 20%
        actual rate = 80%

    False positive (low error):
        estimation rate = 80%
        actual rate = 20%
    '''

    mse = mean_squared_error(
        y.reshape(1, -1),
        y_hat.reshape(1, -1),
        multioutput='raw_values'
    )

    mse = np.where(y_hat >= y, mse * 0.5, mse)
    mse = np.where(y_hat < y, mse * 2, mse)

    return mse.mean()
