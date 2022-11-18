from catboost_ml import CatBoostML, prep_training_inputs
import numpy as np

def main():
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
    catboost_ml = CatBoostML()
    trained_model = catboost_ml.train_regressor(
        features=train_data,
        target=train_targets,
        k_fold_evaluation=True,
        )
    trained_model

if __name__ == '__main__':
    main()