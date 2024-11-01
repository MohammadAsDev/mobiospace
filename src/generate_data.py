import networkx as nx
import numpy as np
from tqdm.auto import tqdm
import pickle
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import settings


"""
    Getting the shortest distance for every landmark
    graph: the graph that we want to work on it
    lanmarks: list of landmarks nodes
    returns a dictionary where key is a landmark node and the value is a list of other nodes' distances 
"""
def get_shortest_distance(graph : nx.Graph, landmarks : list) -> dict:

    distance_map = {}   # this object will be stored as file later 
    nodes = graph.nodes()
    distances = np.zeros((len(nodes), ))    # distances vector

    for landmark in tqdm(landmarks):
        distances[:] = np.inf
        node_dists = nx.shortest_path_length(graph, landmark)
        for key, value in node_dists.items():
            distances[key-1] = value  # since node labels start from 1.
        distance_map[landmark] = distances.copy()  # copy because array is re-init on loop start

    return distance_map

"""
    Get the embeddings for each node
    nodes: list of nodes in the graph
    returns a dictionary where the key is a node identifier and the value is the embeddings for that node.
"""
def get_node_embeddings(nodes : list) -> dict:
    data_emb_path =  settings.DATA_FILES["EMB"]
    try:
        emb_file = open(data_emb_path , "r")
    
    except FileNotFoundError as e:
        print("The graph ({}) is not exist".format(str(data_emb_path)))
        raise e

    emb_data = emb_file.readlines()[1:] 
    
    emb_dict = {}
    for line in emb_data:
        temp = line.split(" ")
        emb_dict[int(temp[0])] = np.array(temp[1:] , dtype=np.float32)
    return emb_dict

"""
    Mix up embeddings
    distance_map: is a dictionary where the key is a landmark node and the value is a list of other nodes' distances
    emb_map: is a dictioanary where the key is a node_id and the value is embeddings for that node
    return a list of pairs (median_embeddings, distance between a landmark and other nodes)
"""
def generate_med_embs(distance_map : dict, emb_map : dict) -> list:
    emb_dist_pairs = []
    for landmark in tqdm(list(distance_map.keys())):
        node_distances = distance_map[landmark]
        for node, distance in enumerate(node_distances, 1):
            if node != landmark and distance != np.inf:
                med = ((emb_map[node] + emb_map[landmark]) / 2 , distance)
                if node == 33:
                    print(f"Node={node} , Land Mark={landmark}, Med={med} , dist={distance}")
                emb_dist_pairs.append(med)
        #emb_dist_pairs.extend(
                #[\
                    #((emb_map[node] + emb_map[landmark]) / 2 , distance)\
                    #for node, distance in enumerate(node_distances , 1)\
                    #if node != landmark and distance != np.inf\
                #])
    return emb_dist_pairs

"""
    using this function, we'll generate two matrices, one for embeddings and the second one for distances
    takes the list of pairs (embeddings, distance) that we've generated using "generate_med_embs"
    return two matrices: embeddings matrix, distances matrix
"""
def generate_train_matricies(emb_dist_pairs : list) -> tuple:
    x = np.zeros((len(emb_dist_pairs) , len(emb_dist_pairs[0][0])))
    y = np.zeros((len(emb_dist_pairs, )))
    
    for i, tup in enumerate(tqdm(emb_dist_pairs)):
        x[i] = tup[0]
        y[i] = tup[1]

    x = x.astype(np.float32)
    y = y.astype(np.int32)

    # remove duplicated nodes

    uniques, idx = np.unique(x , axis = 0, return_index = True)
    dist_set = set(np.arange(0 , len(x)))
    duplicated_nodes = list(dist_set - set(idx)) 
    print("Number of Duplicated Nodes: " , len(duplicated_nodes))
    
    x = x[idx]
    y = y[idx]
    
    #std_scale = MinMaxScaler(feature_range=(0, 1))
    #x = std_scale.fit_transform(x)

   
    # print the final size

    print("x: {} MB\ny: {} MB".format(sys.getsizeof(x)/1024/1024 , sys.getsizeof(y)/1024/1024))
    print("Saved inside: " , str(settings.ROOT_PATH["DUMP"]))
    pickle.dump(x , open(settings.DUMP_FILES["EMB_DATA"] , "wb")) 
    pickle.dump(y , open(settings.DUMP_FILES["DIST_DATA"] , "wb"))
    
    return (x , y)

def main():
    np.random.seed(settings.DEFAULT_SEED)
    edgelist_path = settings.DATA_FILES["GRAPH"] 
    graph = nx.read_edgelist(edgelist_path, nodetype=int)
    # graph = nx.read_weighted_edgelist(edgelist_path, nodetype=int)   # reading ".edgelist" file and store it as graph.

    nodes = [int(i) for i in list(graph.nodes)]     # cast all nodes' ids to "int" instead of "float32"
    landmarks = np.random.randint(1, len(nodes), settings.N_LANDMARKS)   # choosing random 150 landmarks

    distance_map = get_shortest_distance(graph , landmarks) 
    emb_map = get_node_embeddings(nodes)
    
    emb_dist_pairs = generate_med_embs(distance_map , emb_map)

    x, y = generate_train_matricies(emb_dist_pairs)

    


if __name__ == "__main__":
    main()
