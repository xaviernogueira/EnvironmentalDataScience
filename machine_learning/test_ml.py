import numpy as np
import pandas as pd
import catboost as cb
from catboost_ml import CatBoostML

# make dummy data
train_data = np.random.randint(
    0,
    100,
    size=(100, 10),
)
train_targets = np.random.randint(
    0,
    2,
    size=(100),
)
test_data = np.random.randint(
    0,
    100,
    size=(50, 10),
)

df_train = ''
df_targets = ''
df_test = ''

print(f'Train data np array shape: {train_data.shape}')
print(f'Train targets np array shape: {train_targets.shape}')
print(f'Test data np array shape: {test_data.shape}')


def test_catboost_regressor():
    return CatBoostML().train_regressor(
        features=train_data,
        target=train_targets,
        k_fold_evaluation=True,
    )


if __name__ == '__main__':
    test_catboost_regressor()
