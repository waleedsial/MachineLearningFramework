from sklearn import preprocessing
import pandas as pd
import os 
from sklearn import ensemble
from sklearn import metrics
from . import dispatcher
import joblib


# Get the data through environment variables set in run.sh file. 
TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD")) # Fold should be an int 
MODEL = os.environ.get("MODEL")

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

# before we get into the training phase we have already created folds of the data through create folds code. 
# We need to do this only once & depending upon the folds we created, we have the fold mapping here. 



if __name__ == "__main__":
    print("Running Training directly ")

    # READING DATA
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)

    # Fold number is provided through the run.sh file, than we use it to get the fold mapping. for ex 0 will give [1,2,3,4]
    # When the first time it will run it will take the rows which are in the mapping of 0th fold as train & the rest as as validation. 
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))] 
    valid_df = df[df.kfold == FOLD]
    
    # So the idea is that we assigned k folds in the create folds section 
    # Now our data is k folded, each row is assigned some fold number 

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    # We can drop ID column as well.We have the target column separately.we dont need k folds column now. 
    train_df = train_df.drop(["id","target","kfold"], axis = 1)
    valid_df = valid_df.drop(["id","target","kfold"], axis = 1)

    valid_df = valid_df[train_df.columns]

    # Encoding variables 
    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()

        train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
        valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")
        df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")

        lbl.fit(
            train_df[c].values.tolist() + 
            valid_df[c].values.tolist() + 
            df_test[c].values.tolist()
            ) # fitting on training & validation samples 

        train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist()) # Transformation training 
        valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist()) # Transformation Valid set 
        label_encoders[c] = lbl # for each column, we are saving its label encoder in a list. 

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
    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}{FOLD}_.pkl") # We are saving all the models in the model folder. 
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")

else:
    print("Running from Import ")






# if __name__ == "__main__": we use this because sometimes we want to run the code in this file when it is being directly executed where as sometimes
# we want some code in this file to run when it is being imported. https://www.youtube.com/watch?v=sugvnHA7ElY