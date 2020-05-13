import pandas as pd 
from sklearn import model_selection 


if __name__ =="__main__":
    df = pd.read_csv("input/train.csv")
    df["kfold"] = -1


    # Read the data, wholly, remove the index column if there is any
    # Use stratified k fold toc initialize the cross validator object, we are creating folds. 
    df = df.sample(frac = 1).reset_index(drop = True)
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y = df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx , 'kfold'] = fold
        # We mark the dataset rows with their corresponding fold numbers. 

    df.to_csv("input/train_folds.csv", index = False)





    # We provided the full dataset to split function of kf. 
    # Lets say if we are doing 5 fold cross validation 
    # than we split our data into 5 chunks
    # In one iteration we train on 4 of them & test on the left one. 
    # we do this for all the folds. 
    # So if the folds are 5 than loop will run for 5 times. 

    # Split returned the indexes of train & test in each split/fold. 
    # Enumerate is just looping over from 0 to 5 & will vbe treated as fold number. 

    # Fold tells us in how many pieces we are to divide the data. 
    # https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
