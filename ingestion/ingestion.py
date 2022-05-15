import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _ingestion(args):

    sys.path.append("./")

    telecom_cust = pd.read_csv("telecom_churn.csv")
    telecom_cust["TotalCharges"] = telecom_cust["TotalCharges"][telecom_cust["TotalCharges"]!=" "]
    telecom_cust = telecom_cust.dropna()

    # Gets and split dataset
    y = telecom_cust["Churn"].values
    X = telecom_cust.drop(columns = ["Churn"])

    # Creates `data` structure to save and 
    # share train and test datasets.
    data_raw = {"X" : X.values.tolist(), "X_cols": X.columns.tolist(), "y" : y.tolist(), "y_col":["Churn"]}


    # Creates a json object based on `data`
    data_raw = json.dumps(data_raw)

    # Saves the json object into a file
    with open(args.data_raw, "w") as out_file:
        json.dump(data_raw, out_file)

if __name__ == "__main__":
    
    # This component does not receive any input
    # it only outpus one artifact which is `data`.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_raw", type=str)
    
    args = parser.parse_args()
    
    # Creating the directory where the output file will be created 
    # (the directory may or may not exist).
    Path(args.data_raw).parent.mkdir(parents=True, exist_ok=True)

    _ingestion(args)
