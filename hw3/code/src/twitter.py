"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.
    
    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """
    
    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return
    
    np.savetxt(outfile, vec)    


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    count = 0
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        for line in fid:
            unique_words = extract_words(line)
            for word in unique_words:
                if word not in word_list:
                    word_list[word] = count
                    count += 1
        pass
        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        for line_num, line in enumerate(fid):
            words = extract_words(line)
            for word in words:
                feature_matrix[line_num][word_list[word]] = 1
                
        pass
        ### ========== TODO : END ========== ###
        
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_label)
    elif metric == "f1-score":
        return metrics.f1_score(y_true, y_label, average='binary')
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_label)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_label, average='binary')
    elif metric == "sensitivity":
        confusion = metrics.confusion_matrix(y_true, y_label)
        TN, FP    = confusion[0, 0], confusion[0, 1]
        FN, TP    = confusion[1, 0], confusion[1, 1]
        return TP / float(TP + FN)
    elif metric == "specificity":
        confusion = metrics.confusion_matrix(y_true, y_label)
        TN, FP    = confusion[0, 0], confusion[0, 1]
        FN, TP    = confusion[1, 0], confusion[1, 1]
        return TN / float(TN + FP)
    return 0
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance  
    scores = []
    for train_index, test_index in kf:
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.decision_function(X_test)
        scores.append(performance(y_test, y_pred, metric=metric))
    # print(scores)
    return np.mean(scores)
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2c: select optimal hyperparameter using cross-validation
    max_perf = 0
    C = None
    for i in C_range:
        clf = SVC(kernel='linear', C=i)
        perf = cv_performance(clf, X, y, kf, metric=metric)
        print("C = " + str(i) + ": " + str(perf))
        if perf > max_perf:
            C = i
            max_perf = perf
    return C
    ### ========== TODO : END ========== ###


def select_param_rbf(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """
    
    print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ':'
    
    ### ========== TODO : START ========== ###
    # part 3b: create grid, then select optimal hyperparameters using cross-validation
    C_range = 10.0 ** np.arange(-3, 3)
    Gamma_range = 10.0 ** np.arange(-3, 3)

    max_perf = 0
    C = None
    G = -1

    for c in C_range:
        for g in Gamma_range:
            clf = SVC(kernel='rbf', C=c, gamma=g)
            perf = cv_performance(clf, X, y, kf, metric=metric)
            if perf > max_perf:
                C = c
                G = g
                max_perf = perf

    print("C = " + str(C) + ", gamma = " + str(G) + ": " + str(max_perf))
    return G, C
    ### ========== TODO : END ========== ###


def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 4b: return performance on test data by first computing predictions and then calling performance
    y_pred = clf.decision_function(X)
    return performance(y, y_pred, metric=metric)
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')
    # print(dictionary)
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')
    
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    
    ### ========== TODO : START ========== ###
    # part 1c: split data into training (training + cross-validation) and testing set
    MAX_TRAINING_ROWS = 560
    train_feature = X[:MAX_TRAINING_ROWS]
    train_label = y[:MAX_TRAINING_ROWS]
    test_feature = X[MAX_TRAINING_ROWS:]
    test_label = y[MAX_TRAINING_ROWS:]
    
    # y_pred = [0.9, -1.0, 1.0, 0.8]
    # y_true = [1, 1, -1, 1]

    # print(performance(y_true, y_pred))
    # print(performance(y_true, y_pred, metric='f1-score'))
    # print(performance(y_true, y_pred, metric='auroc'))
    # print(performance(y_true, y_pred, metric='precision'))
    # print(performance(y_true, y_pred, metric='sensitivity'))
    # print(performance(y_true, y_pred, metric='specificity'))

    # part 2b: create stratified folds (5-fold CV)
    # clf = SVC(kernel='linear', C=10**-2)
    # kf = StratifiedKFold(train_label, n_folds=5)
    # print(cv_performance(clf, X, y, kf))

    # part 2d: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    metrics = ['accuracy', 'f1-score', 'auroc', 'precision', 'sensitivity', 'specificity']
    # for m in metrics:
        # print('Best c:    ' + str(select_param_linear(train_feature, train_label, kf, metric=m)))

    # part 3c: for each metric, select optimal hyperparameter for RBF-SVM using CV
    # for m in metrics:
        # print(select_param_rbf(train_feature, train_label, kf, metric=m))


    # part 4a: train linear- and RBF-kernel SVMs with selected hyperparameters
    lin_clf = SVC(kernel='linear', C=10)
    lin_clf.fit(train_feature, train_label)

    rbf_clf = SVC(kernel='rbf', C=100, gamma=0.01)
    rbf_clf.fit(train_feature, train_label)


    # part 4c: report performance on test data
    for m in metrics:
        score = performance_test(lin_clf, test_feature, test_label, metric=m)
        print("Linear Metric: " + m + " | Score: " + str(score))

        score = performance_test(rbf_clf, test_feature, test_label, metric=m)
        print("RBF Metric: " + m + " | Score: " + str(score))
    
    ### ========== TODO : END ========== ###
    
    
if __name__ == "__main__" :
    main()
