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
    # mu = [[0.75,0.75], [1,1], [1.25,1.25]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]
    # sigma = [[0.001,0.1], [0.014,0.01], [0.013,0.1]]
    
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
    # return np.random.choice(points, k, replace=False)
    indices = np.random.choice(len(points),k,replace=False)
    initial_points = []
    for index in indices:
        initial_points.append(points[index])
    return initial_points

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
    initial_points = {}
    for p in points:
        if p.label not in initial_points:
            initial_points[p.label] = [p]
        else:
            initial_points[p.label].append(p)

    result = []
    for label in initial_points:
        sorted_points = initial_points[label]
        c = Cluster(sorted_points)
        result.append(c.medoid())

    return result
    ### ========== TODO : END ========== ###

def kAverages(points, k, average, init='random', plot=False):
    # initialize ClusterSet
    k_clusters = ClusterSet()
    if init == 'random':
        init_pts = random_init(points, k)
        for i_pt in init_pts:
            k_clusters.add(Cluster([i_pt]))
    elif init == 'cheat':
        for m in cheat_init(points):
            k_clusters.add(Cluster([m]))

    averages = average(k_clusters)
    # for c in averages:
    #     print(str(c))
    # Make another ClusterSet to compare with k_clusters after one iteration of kmeans
    new_k_clusters = ClusterSet()

    iteration_num = 1
    while not k_clusters.equivalent(new_k_clusters): 
        updated_cluster_points = [[] for x in xrange(k)]
        
        # Set k_clusters to the result of the previous iteration of kmeans
        k_clusters = ClusterSet()
        for cluster in new_k_clusters.members:
            k_clusters.add(cluster)

        # Reset new_k_clusters for the following iteration
        new_k_clusters = ClusterSet()

        for p in points:
            dist_from_cluster = []

            for center in averages:
                dist_from_cluster.append(p.distance(center))

            # find cluster which has closest centroid from this point
            cluster_index = np.argmin(dist_from_cluster)
            updated_cluster_points[cluster_index].append(p)

        for i, c in enumerate(averages):
            # if len(updated_cluster_points[i]) == 0:
            #     print('empty!')
            #     continue
            new_k_clusters.add(Cluster(updated_cluster_points[i]))

        # print(new_k_clusters)
        averages = average(new_k_clusters)
        # print('==========')
        # for c in averages:
        #     print(str(c))
        # PLOT
        if plot is True:
            plot_clusters(new_k_clusters, str(average) + " iteration #" + str(iteration_num), average)
        iteration_num += 1

    return k_clusters

def kMeans(points, k, init='random', plot=False):
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

    return kAverages(points, k, ClusterSet.centroids, init=init, plot=plot)
    ### ========== TODO : END ========== ###

def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part 2e: implement
    return kAverages(points, k, ClusterSet.medoids, init=init, plot=plot)
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
    
    # 1b
    # U, mu = util.PCA(X)
    # n,d = U.shape
    # plot_gallery([vec_to_image(U[:,i]) for i in xrange(12)])
    # for column_index in range(d):
    #     col = U[:,column_index]
    #     util.show_image(util.vec_to_image(col))
    
    # 1c
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
    # np.random.seed(1234)
    # k = 3
    # pts_per_cluster = 20
    # for i in range(1):
    #     points = generate_points_2d(pts_per_cluster)
    #     k_clusters = kMeans(points, k, init="cheat", plot=True)
    #     k_clusters = kMedoids(points, k, init="cheat", plot=True)
    ### ========== TODO : END ========== ###
    
    ### ========== TODO : START ========== ###    
    # part 3a: cluster faces
    np.random.seed(1234)
    k = 4
    X1, y1 = util.limit_pics(X, y, [4, 6, 13, 16], 40)
    points = build_face_image_points(X1, y1)

    # plot = {}
    # for pt in points:
    #     if pt.label not in plot:
    #         plot[pt.label] = []
    #     plot[pt.label].append(pt)
    # clusters = ClusterSet()
    # for l in plot:
    #     clusters.add(Cluster(plot[l]))
    # plot_clusters(clusters, 'orig', ClusterSet.centroids)

    # Part 3a
    # centroid_score = []
    # medoid_score = []
    # for i in range(10):
    #     k_clusters = kMeans(points, k, init="random", plot=False)
    #     centroid_score.append(k_clusters.score())

    # centroid_mean = sum(centroid_score) / float(len(centroid_score))
    # centroid_min = min(centroid_score)
    # centroid_max = max(centroid_score)
    # print('Centroid avg:', centroid_mean)
    # print('Centroid min:', centroid_min)
    # print('Centroid max:', centroid_max)

    # medoid_score = []
    # for i in range(10):
    #     k_clusters = kMedoids(points, k, init="random", plot=False)
    #     medoid_score.append(k_clusters.score())

    # centroid_mean = sum(medoid_score) / float(len(medoid_score))
    # centroid_min = min(medoid_score)
    # centroid_max = max(medoid_score)
    # print('Medoid avg:', centroid_mean)
    # print('Medoid min:', centroid_min)
    # print('Medoid max:', centroid_max)

    # PART 3b
    X1, y1 = util.limit_pics(X, y, [4, 13], 40)
    U, mu = util.PCA(X1)
    k = 2
    ls = range(42)[1::2]

    centroid_score = []
    medoid_score = []

    for l in ls:
        Z, Ul = util.apply_PCA_from_Eig(X1, U, l, mu)
        X_rec = util.reconstruct_from_PCA(Z, Ul, mu)
        points = build_face_image_points(Z, y1)
        # plot_gallery(X_rec[:12])

        c = kMeans(points, k, init="cheat", plot=False)
        centroid_score.append(c.score())
        k_clusters = kMedoids(points, k, init="cheat")
        medoid_score.append(k_clusters.score())

    scatter = plt.scatter(ls, centroid_score, c='c', s=20)
    scatter2 = plt.scatter(ls, medoid_score, c='r', s=20)
    plt.suptitle('kMeans and kMedoids', fontsize=20)
    plt.xlabel('L', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.legend((scatter, scatter2),
               ('kMeans', 'kMedoids'),
               scatterpoints=1,
               loc='lower right',
               ncol=3,
               fontsize=14)
    plt.show()

    # part 3b: explore effect of lower-dimensional representations on clustering performance
    np.random.seed(1234)
    
    # part 3c: determine ``most discriminative'' and ``least discriminative'' pairs of images
    np.random.seed(1234)
    
    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
