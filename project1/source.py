# CS 565 Data mining Project 1
# Tian Zhang U31287117
# Using Kmeans and Kmeans++ to do clustering among movie dataset

#Usage:
# python source.py -d <path to file> -c <clusterNum> -init <method to use>
# "random" for Kmeans
# "k-means++" for Kmeans plus plus

# Reference:
# 1. https://github.com/jackmaney/k-means-plus-plus-pandas
# 2. http://brandonrose.org/clustering
# 3. http://kenzotakahashi.github.io/k-means-clustering-from-scratch-in-python.html
# https://github.com/Behrouz-Babaki

import argparse as ap
import numpy as np
import pandas as pd
import random
import json

# Func for data normalization
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

# Some categorical feature is stored in JSON string, we need to parse it to array.
def parsingJSON(featureName):
    featureAll = []
    for k,x in enumerate(featureName):
        featureList = []
        feature = json.loads(featureName[k])
        for i,val in enumerate(feature):
            featureList.append(feature[i]['name'])
        featureAll.append(featureList)
    return featureAll

# taken from scikit-learn (https://goo.gl/1RYPP5)
def tolerance(tol, dataset):
    n = len(dataset)
    dim = len(dataset[0])
    averages = [sum(dataset[i][d] for i in range(n))/float(n) for d in range(dim)]
    variances = [sum((dataset[i][d]-averages[d])**2 for i in range(n))/float(n) for d in range(dim)]
    return tol * sum(variances) / dim

def closest_clusters(centers, datapoint):
    distances = [l2_distance(center, datapoint) for
                 center in centers]
    return sorted(range(len(distances)), key=lambda x: distances[x]), distances

# Defined two method for initial center choosing. 
# "1" for kmeans
# "2" for kmeans++
def initialize_centers(dataset, k, method):
    if method == 'random':
        ids = list(range(len(dataset)))
        random.shuffle(ids)
        return [dataset[i] for i in ids[:k]]        
    
    elif method == 'k-means++”':
        chances = [1] * len(dataset)
        centers = []
        
        for _ in range(k):
            chances = [x/sum(chances) for x in chances]        
            r = random.random()
            acc = 0.0
            for index, chance in enumerate(chances):
                if acc + chance >= r:
                    break
                acc += chance
            centers.append(dataset[index])
            
            for index, point in enumerate(dataset):
                cids, distances = closest_clusters(centers, point)
                chances[index] = distances[cids[0]]
                
        return centers

def violate_constraints(data_index, cluster_index, clusters, ml, cl):
    for i in ml[data_index]:
        if clusters[i] != -1 and clusters[i] != cluster_index:
            return True

    for i in cl[data_index]:
        if clusters[i] == cluster_index:
            return True

    return False

# Used L2^2 for distance function
def l2_distance(point1, point2):
    return np.sqrt(sum([(float(i)-float(j))**2 for (i, j) in zip(point1, point2)]))

# Kmeans Func
def kmeans(dataset, k, initialization, max_iter=300, tol=1e-2):
    ml=[]
    cl=[]
    ml, cl = transitive_closure(ml, cl, len(dataset))
    ml_info = get_ml_info(ml, dataset)
    tol = tolerance(tol, dataset)
    
    centers = initialize_centers(dataset, k, initialization)
    clusters = [-1] * len(dataset)

    for i in range(max_iter):
        clusters_ = [-1] * len(dataset)
        for i, d in enumerate(dataset):
            indices, _ = closest_clusters(centers, d)
            counter = 0
            if clusters_[i] == -1:
                found_cluster = False
                while (not found_cluster) and counter < len(indices):
                    index = indices[counter]
                    if not violate_constraints(i, index, clusters_, ml, cl):
                        found_cluster = True
                        clusters_[i] = index
                        for j in ml[i]:
                            clusters_[j] = index
                    counter += 1

                if not found_cluster:
                    return None
        
        clusters_, centers_ = compute_centers(clusters_, dataset, k, ml_info)
        shift = sum(l2_distance(centers[i], centers_[i]) for i in range(k))
        if shift <= tol:
            break
        
        clusters = clusters_
        centers = centers_
        inertia_ = sum(((centers[l] - x)**2).sum()
                            for x, l in zip(dataset, clusters))

    return clusters_, centers_, inertia_

def compute_centers(clusters, dataset, k, ml_info):
    cluster_ids = set(clusters)
    k_new = len(cluster_ids)
    id_map = dict(zip(cluster_ids, range(k_new)))
    clusters = [id_map[x] for x in clusters]    
    
    dim = len(dataset[0])
    centers = [[0.0] * dim for i in range(k)]

    counts = [0] * k_new
    for j, c in enumerate(clusters):
        for i in range(dim):
            centers[c][i] += dataset[j][i]
        counts[c] += 1
        
    for j in range(k_new):
        for i in range(dim):
            centers[j][i] = centers[j][i]/float(counts[j])

    if k_new < k:
        ml_groups, ml_scores, ml_centroids = ml_info
        current_scores = [sum(l2_distance(centers[clusters[i]], dataset[i]) 
                              for i in group) 
                          for group in ml_groups]
        group_ids = sorted(range(len(ml_groups)), 
                           key=lambda x: current_scores[x] - ml_scores[x],
                           reverse=True)
        
        for j in range(k-k_new):
            gid = group_ids[j]
            cid = k_new + j
            centers[cid] = ml_centroids[gid]
            for i in ml_groups[gid]:
                clusters[i] = cid
                
    return clusters, centers
    
def get_ml_info(ml, dataset):
    flags = [True] * len(dataset)
    groups = []
    for i in range(len(dataset)):
        if not flags[i]: continue
        group = list(ml[i] | {i})
        groups.append(group)
        for j in group:
            flags[j] = False
    
    dim = len(dataset[0])
    scores = [0.0] * len(groups)
    centroids = [[0.0] * dim for i in range(len(groups))]
    
    for j, group in enumerate(groups):
        for d in range(dim):
            for i in group:
                centroids[j][d] += dataset[i][d]
            centroids[j][d] /= float(len(group))

    scores = [sum(l2_distance(centroids[j], dataset[i])
                  for i in groups[j]) 
              for j in range(len(groups))]
    
    return groups, scores, centroids

def transitive_closure(ml, cl, n):
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in cl:
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' %(i, j))

    return ml_graph, cl_graph

if __name__ == "__main__":
    # # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-d', "--dataset", help="Path to the dataset", required=True)
    parser.add_argument(
        '-c', '--cluster', help="Enter the quantity of cluster", required=True, type=int)
    parser.add_argument('-init', '--initialization',
                        help="Enter a method you want to use for clustering", required=True)
    args = vars(parser.parse_args())

    clusterNum = args["cluster"]
    method = args["initialization"]

    movie = pd.read_csv(args["dataset"])

    # Using a dataframe 'movie_num' to store chosen numerical columns
    num_list = ['budget','popularity','revenue','vote_average'] 
    movie_num = movie[num_list] 

    # Normalze the numerical features chosen above
    movieNum = normalize(movie_num)

    # Parsing categorical features, including genres, keywords, production_companies, production_countries and spoken_languages.
    movie.genres = parsingJSON(movie.genres)
    movie.keywords = parsingJSON(movie.keywords)
    movie.production_companies = parsingJSON(movie.production_companies)
    movie.production_countries = parsingJSON(movie.production_countries)
    movie.spoken_languages = parsingJSON(movie.spoken_languages)

    # Spliting categorical features to dummy variables
    genreDummies = movie.genres.astype(str).str.strip('[]').str.get_dummies(', ')
    keywordDummies = movie.keywords.astype(str).str.strip('[]').str.get_dummies(', ')
    production_companyDummies = movie.production_companies.astype(str).str.strip('[]').str.get_dummies(', ')
    production_countryDummies = movie.production_countries.astype(str).str.strip('[]').str.get_dummies(', ')
    spoken_languageDummies = movie.spoken_languages.astype(str).str.strip('[]').str.get_dummies(', ')

    # Merging numerical features for clustering
    finalDF = pd.merge(movieNum,genreDummies,how='inner',left_index=True, right_index=True)
    X = finalDF.as_matrix()
    cluster, center, inertia = kmeans(X,clusterNum,method)
    movie['label'] = cluster
    result = ['id','label']
    resultDF = movie[result]
    resultDF.to_csv("output.csv")
	

