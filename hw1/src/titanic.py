"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set

        num_survived = float(np.sum(y == 1.))
        total_passengers = len(y)
        self.probabilities_ = { 
            'survived' : num_survived / total_passengers,
            'not_survived' : (total_passengers - num_survived) / total_passengers
        }
        
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)

        y = np.random.choice([0,1], size=len(X), p=[self.probabilities_['not_survived'], self.probabilities_['survived']])
        
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in xrange(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in xrange(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    
    train_error = 0
    test_error = 0    

    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)

        # Learn model based on split data
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        train = 1 - metrics.accuracy_score(y_train, y_train_pred, normalize=True)
        test = 1 - metrics.accuracy_score(y_test, y_test_pred, normalize=True)
        
        train_error += train
        test_error += test 

    ### ========== TODO : END ========== ###
    
    return train_error / ntrials, test_error / ntrials


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    # print 'Plotting...'
    # for i in xrange(d) :
    #     plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    #========================================
    # train Majority Vote classifier on data
    print 'Classifying using Majority Vote...'
    majority_clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    majority_clf.fit(X, y)                  # fit training data using the classifier
    y_pred = majority_clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print 'Classifying using Random...'
    random_clf = RandomClassifier()
    random_clf.fit(X, y)
    y_pred = random_clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print 'Classifying using Decision Tree...'
    decision_tree_clf = DecisionTreeClassifier(criterion='entropy')
    decision_tree_clf.fit(X, y)
    y_pred = decision_tree_clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error
    
    
    ### ========== TODO : END ========== ###
    
    
    
    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """
    
    
    
    ### ========== TODO : START ========== ###
    # part d: use cross-validation to compute average training and test error of classifiers
    print 'Investigating various classifiers...'
    # majority_err_train, majority_err_test, random_err_train, random_err_test, decision_tree_err_train, decision_tree_err_test = 0
    
    majority_err_train, majority_err_test = error(majority_clf, X, y)
    random_err_train, random_err_test = error(random_clf, X, y)
    decision_tree_err_train, decision_tree_err_test = error(decision_tree_clf, X, y)

    print('majority_clf:', majority_err_train, majority_err_test)
    print('random_clf:', random_err_train, random_err_test)
    print('decision_tree_clf:', decision_tree_err_train, decision_tree_err_test)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: investigate decision tree classifier with various depths
    print 'Investigating depths...'
    # depths = range(1,21)
    # training_errors = []
    # test_errors = []
    # for i in depths:
    #     decision_tree_depth_clf = DecisionTreeClassifier(criterion='entropy', max_depth=i)
    #     training_depth_error, test_depth_error = error(decision_tree_depth_clf, X, y)
    #     training_errors.append(training_depth_error)
    #     test_errors.append(test_depth_error)



    # # Make plot
    # train_scatter = plt.scatter(depths, training_errors, c='c', s=35)
    # test_scatter= plt.scatter(depths, test_errors, c='r', s=35)
    # majority_scatter = plt.plot((0, 20), (0.40734265734265768, 0.40734265734265768), 'k-')
    # random_scatter = plt.plot((0, 20), (0.48657342657342667, 0.48657342657342667), 'b')
    # plt.suptitle('Decision Tree Depths Affect on Error', fontsize=20)
    # plt.xlabel('max depth', fontsize=18)
    # plt.ylabel('error', fontsize=16)
    # plt.legend((train_scatter, test_scatter, majority_scatter, random_scatter),
    #            ('training data', 'testing data'),
    #            scatterpoints=1,
    #            loc='lower left',
    #            ncol=2,
    #            fontsize=14)

    # plt.show()
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part f: investigate decision tree classifier with various training set sizes
    print 'Investigating training set sizes...'
    best_tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    best_training_errors = []
    best_test_errors = []
    sizes = np.arange(0.05, 1, 0.05)
    for split in sizes:
        best_training_error, best_test_error = error(best_tree_clf, X, y, test_size=1-split)
        best_training_errors.append(best_training_error)
        best_test_errors.append(best_test_error)
    
    train_scatter = plt.scatter(sizes, best_training_errors, c='c', s=35)
    test_scatter= plt.scatter(sizes, best_test_errors, c='r', s=35)
    
    plt.suptitle('Decision Tree Training Set Size Affect on Error', fontsize=20)
    plt.xlabel('training data size', fontsize=18)
    plt.ylabel('error', fontsize=16)
    plt.legend((train_scatter, test_scatter),
               ('training data', 'testing data'),
               scatterpoints=1,
               loc='upper left',
               ncol=3,
               fontsize=14)

    plt.show()

    ### ========== TODO : END ========== ###
    
       
    print 'Done'


if __name__ == "__main__":
    main()
