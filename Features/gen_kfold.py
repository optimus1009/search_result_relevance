
"""
__file__

    gen_kfold.py

__description__

    This file generates the StratifiedKFold indices which will be kept fixed in
    ALL the following model building parts.

__author__

    xionglin <optimus1009@163.com>

"""

import sys
import cPickle
from sklearn.cross_validation import StratifiedKFold
sys.path.append("../")
from param_config import config


if __name__ == "__main__":

    ## load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = cPickle.load(f)
    
    ## CV params
    ## self.n_runs = 3
    ## self.n_folds = 3
    ## self.stratified_label = "query"
    skf = [0]*config.n_runs
    ## skf = [0,0,0]
    for stratified_label,key in zip(["relevance", "query"], ["median_relevance", "qid"]):
        for run in range(config.n_runs):
            random_seed = 2015 + 1000 * (run+1)
            skf[run] = StratifiedKFold(dfTrain[key], n_folds=config.n_folds,
                                        shuffle=True, random_state=random_seed)
            for fold, (validInd, trainInd) in enumerate(skf[run]):
                print("================================")
                print("Index for run: %s, fold: %s" % (run+1, fold+1))
                print("Train (num = %s)" % len(trainInd))
                print(trainInd[:10])
                print("Valid (num = %s)" % len(validInd))
                print(validInd[:10])
        with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, stratified_label), "wb") as f:
            cPickle.dump(skf, f, -1)