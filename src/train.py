from sklearn import preprocessing
import pandas as pd
import os 
from sklearn import ensemble
from sklearn import metrics
from . import dispatcher
import joblib


# We ned FOLD & FOLD Mapping, Model trhough environment variables 
TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD")) # Fold should be an int 
FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}
MODEL = os.environ.get("MODEL")




if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))] # here k fold is in a list 
    valid_df = df[df.kfold == FOLD]
    # So the idea is that we assigned k folds in the create folds section 
    # Now our data is k folded, each row is assigned some fold number 

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    # We have the target column separately
    # we dont need k folds column now. 
    # We can drop ID column as well.
    train_df = train_df.drop(["id","target","kfold"], axis = 1)
    valid_df = valid_df.drop(["id","target","kfold"], axis = 1)


    # Encoding variables 
    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist()) # fitting on training & validation samples 
        train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist()) # Transformation training 
        valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist()) # Transformation Valid set 
        label_encoders.append((c,lbl)) # Appending column name & lable encoder 

    # Now our data is ready to train 

    clf = dispatcher.MODELS[MODEL] # We got the model name from env variable, than we passed it to dispatcher, which will returnd the selected intialized model. 
    # Initially our estimators are set at 100, we can increase it to see if accuracy imprvs. 
    # n_estimators = 100 got us 0.7359758293464235
    # with 200 estimators , score is 0.7414975823473439
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    # print(preds)
    # When data is skewed roc is a better way to check our modelling 
    print(metrics.roc_auc_score(yvalid, preds))

    # Saving Results 
    # Dump is saving the python objects into some files. PKL is a file format for effective serialization & deserialization 
    # joblib.dump takes python object that we want to save & the filename
    # So we are saving label encoders & models themselves
    # Why do we need to save the label encoders ? 
    joblib.dump(label_encoders, f"models/{MODEL}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}.pkl") # We are saving all the models in the model folder. 





