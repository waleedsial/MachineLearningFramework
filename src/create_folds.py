import pandas as pd 
from sklearn import model_selection 


if __name__ =="__main__":
    df = pd.read_csv("input/train.csv")
    df["kfold"] = -1

    df = df.sample(frac = 1).reset_index(drop = True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42)


    # So here basically we are iterating over the folds. Enumerate returns the fold & indexes
    # kf.split takes data & splits it into training & test indexes. 
    # so we passed in the X values & Y values & resultant will be train & test indexes for each split. 
    # by default it will create 5 splits.
    #  Split function will return the indices in a list. Which we can interate over to find the indexes of the splitted data. 
    #  So for example, if there were 5 splits than it will have 5 entries in each return list. 
    # so for 
    # val_idx is the left out set for the k fold. 
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y = df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx , 'kfold'] = fold


    df.to_csv("input/train_folds.csv", index = False)

