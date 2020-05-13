from sklearn import preprocessing
import pandas as pd
import os 
from sklearn import ensemble
from sklearn import metrics
from . import dispatcher
import joblib
import numpy as np 


# TEst data, Model trhough environment variables 

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["id"].values
    predictions = None

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(TEST_DATA) # Since we are changing the data, therefore we read in every time we come in fold. 
        encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
        for c in encoders:
            lbl = encoders[c]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            df.loc[:,c] = lbl.transform(df[c].values.tolist()) # We just need transformation in prediction.


        clf = joblib.load(os.path.join("models", f"{MODEL}{FOLD}_.pkl"))

        df = df[cols]
        preds = clf.predict_proba(df)[:,1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns= ["id","target"])
    return sub

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index = False)






