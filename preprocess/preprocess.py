import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _preprocess(args):

    # Open and reads file "data"
    with open(args.data_raw) as data_file:
        data_raw = json.load(data_file)
    
    # The excted data type is 'dict', however since the file
    # was loaded as a json object, it is first loaded as a string
    # thus we need to load again from such string in order to get 
    # the dict-type object.
    data = json.loads(data_raw)

    X = data['X']
    y = data['y']

    X = pd.DataFrame(data['X'], columns=data["X_cols"])
    y = pd.DataFrame(data['y'], columns=data["y_col"])


    # Converting Total Charges to a numerical data type.
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"])

    # Remove customer IDs from the data set
    X = X.iloc[:,1:]

    # Convertin the predictor variable in a binary numeric variable
    y['Churn'].replace(to_replace='Yes', value=1, inplace=True)
    y['Churn'].replace(to_replace='No',  value=0, inplace=True)

    # Let's convert all the categorical variables into dummy variables
    X = pd.get_dummies(X)

    # Split Dataset to train/test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # # Initialize and train the model
    data = {"X_train" : x_train.values.tolist(), "X_train_cols": x_train.columns.tolist(), "y_train" : y_train.values.tolist(),
            "X_test" : x_test.values.tolist(), "X_test_cols": x_test.columns.tolist(), "y_test" : y_test.values.tolist()}

    data = json.dumps(data)

    # Save output into file
    with open(args.data, "w") as out_file:
        json.dump(data, out_file)



if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Preprocessing Train')
    parser.add_argument('--data_raw', type=str)
    parser.add_argument('--data', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created 
    # (the directory may or may not exist).
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)
    
    _preprocess(args)
