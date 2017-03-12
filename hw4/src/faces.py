"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Famous Faces
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
import util
from util import *
from cluster import *

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets
    
    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """
    
    n,d = X.shape
    
    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])
    
    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """
    
    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.
    
    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed
    
    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)
    
    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]
    
    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))
    
    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2c: implement (hint: use np.random.choice)
    return np.random.choice(points, k, replace=False)
    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!
    
    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2f: implement
    initial_points = []
    return initial_points
    ### ========== TODO : END ========== ###


def kMeans(points, k, init='random', plot=False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.
    
    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable: 
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm
    
    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """
    
    ### ========== TODO : START ========== ###
    # part 2c: implement
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to create
    #       a new ClusterSet object and update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).

    # initialize ClusterSet
    k_clusters = ClusterSet()
    for cluster in range(k):
        if init == 'random':
            k_clusters.add(Cluster(random_init(points, 1)))
        elif init == 'cheat':
            k_clusters.add(Cluster(cheat_init(points)))

    centroids = k_clusters.centroids()

    # Make another ClusterSet to compare with k_clusters after one iteration of kmeans
    new_k_clusters = ClusterSet()

    while not k_clusters.equivalent(new_k_clusters):  
        updated_cluster_points = [[] for x in xrange(len(centroids))]
        
        # Set k_clusters to the result of the previous iteration of kmeans
        k_clusters = ClusterSet()
        for cluster in new_k_clusters.members:
            k_clusters.add(cluster)

        # Reset new_k_clusters for the following iteration
        new_k_clusters = ClusterSet()

        for p in points:
            dist_from_cluster = []

            for center in centroids:
                dist_from_cluster.append(p.distance(center))
            
            # missing paramater in func signature?
            # medoids = k_clusters.medoids()

            # find cluster which has closest centroid from this point
            cluster_index = np.argmin(dist_from_cluster)
            updated_cluster_points[cluster_index].append(p)

        for i, c in enumerate(centroids):
            new_k_clusters.add(Cluster(updated_cluster_points[i]))

        centroids = new_k_clusters.centroids()

    if plot is True:
        plot_clusters(k_clusters, "kMeans with " + str(k) + " clusters", ClusterSet.centroids)

    return k_clusters
    ### ========== TODO : END ========== ###


def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part 2e: implement
    k_clusters = ClusterSet()
    return k_clusters
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    ### ========== TODO : START ========== ###
    # part 1: explore LFW data set
    X, y = util.get_lfw_data()
    # n,d = X.shape
    # avg_face = []
    # for column_index in range(d):
    #     col = X[:,column_index]
    #     avg_face_attr = np.mean(col, axis=0)
    #     avg_face.append(avg_face_attr)

    # util.show_image(np.array(avg_face))
    ### ========== TODO : END ========== ###
    
    # 2b
    # U, mu = util.PCA(X)
    # n,d = U.shape
    # plot_gallery([vec_to_image(U[:,i]) for i in xrange(12)])
    # for column_index in range(d):
    #     col = U[:,column_index]
    #     util.show_image(util.vec_to_image(col))
    
    # 2c
    # ls = [1, 10, 50, 100, 500, 1288]
    # for l in ls:
    #     Z, Ul = util.apply_PCA_from_Eig(X, U, l, mu)
    #     X_rec = util.reconstruct_from_PCA(Z, Ul, mu)
    #     plot_gallery(X_rec[:12])
    

    # test centroid
    # p1 = Point('1', 1, np.array([5, 4]))
    # p2 = Point('2', 2, np.array([9, 10]))
    # p3 = Point('3', 3, np.array([3, 9]))
    # c = Cluster([p1, p2, p3])
    # print(str(c))
    # print(str(c.centroid()))
    # end test centroid

    ### ========== TODO : START ========== ###
    # part 2d-2f: cluster toy dataset
    np.random.seed(1235)
    k = 3
    pts_per_cluster = 20
    for i in range(10):
        np.random.seed(i * 1000)
        points = generate_points_2d(pts_per_cluster * k, seed=np.random.random_integers(100000))
        k_clusters = kMeans(points, k, plot=True)
    # print(str(k_clusters))
    # print('*****************************************')
    # for cluster in k_clusters.members:
        # print(str(cluster.centroid()))

    # for cluster in k_clusters:
    #     print(str(cluster))
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###    
    # part 3a: cluster faces
    np.random.seed(1234)
        
    # part 3b: explore effect of lower-dimensional representations on clustering performance
    np.random.seed(1234)
    
    # part 3c: determine ``most discriminative'' and ``least discriminative'' pairs of images
    np.random.seed(1234)
    
    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
