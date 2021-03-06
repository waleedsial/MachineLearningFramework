from sklearn import ensemble

# Dispatcher is a file which will have different models. 
# We initiate the model here. Using command line we can choose which model we want.
# The rest of the methods like preparing the data, fitting the model are similar/same for different models. 

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators = 200,n_jobs = -1, verbose =2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators = 200,n_jobs = -1, verbose =2)
}

# These variables are passed through run.sh , we tell through command line which model we want to run. 