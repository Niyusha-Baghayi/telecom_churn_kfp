import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def _random_forest(args):

    # Open and reads file "data"
    with open(args.data) as data_file:
        data = json.load(data_file)
    
    # The excted data type is 'dict', however since the file
    # was loaded as a json object, it is first loaded as a string
    # thus we need to load again from such string in order to get 
    # the dict-type object.
    data = json.loads(data)

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']


    X_train = pd.DataFrame(X_train, columns=data["X_train_cols"])
    X_test = pd.DataFrame(X_test, columns=data["X_test_cols"])



    # Initialize and train the model
    model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                    random_state =50, max_features = "auto",
                                    max_leaf_nodes = 30)
    model_rf.fit(X_train, y_train)

    # Get predictions
    y_pred = model_rf.predict(X_test)

    # Get accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print the prediction accuracy
    print (accuracy)

    importances = model_rf.feature_importances_
    weights = pd.Series(importances,
                    index=X_train.columns.values)
    print(weights.sort_values()[-10:].plot(kind = 'barh'))

    # Save output into file
    with open(args.accuracy, 'w') as accuracy_file:
        accuracy_file.write(str(accuracy))



if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Random Forest')
    parser.add_argument('--data', type=str)
    parser.add_argument('--accuracy', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.accuracy).parent.mkdir(parents=True, exist_ok=True)
    
    _random_forest(args)
