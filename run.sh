export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv


export MODEL=$1
# FOLD=0 python -m src.train
# FOLD=1 python -m src.train
# FOLD=2 python -m src.train
# FOLD=3 python -m src.train
# FOLD=4 python -m src.train
python -m src.predict













#  We have an SH file for running the training. 
# Seeting environment variables & calling the train file 
# sh files are unix (linux) shell executables files, they are the equivalent (but much more powerful) of bat files on windows.
# So you need to run it from a linux console, just typing its name the same you do with bat files on windows.
# https://stackoverflow.com/questions/13805295/whats-a-sh-file
# Run sh 


# Exported variables such as $HOME and $PATH are available to (inherited by) other programs run by the shell that exports them 
# (and the programs run by those other programs, and so on) as environment variables. 
# Dont use space in bash 