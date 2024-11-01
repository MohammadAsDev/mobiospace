from sklearn.svm import SVC
import torch as tch
import torch.nn as nn

import numpy as np

import networkx as nx 

import pickle

import random

import settings
import math

def poisson_loss(y_pred , y_true):
    y_pred = tch.squeeze(y_pred)
    loss = tch.mean(y_pred - y_true * tch.log(y_pred+1e-7))
    return loss

def test(model, loss_fn, data_loader):
    model.eval()
    final_loss = 0.0
    count = 0
    y_hat = []
    with tch.no_grad():
        for data in data_loader:
            inputs, dist_true = data[0] , data[1]
            count += len(inputs)
            outputs = model(inputs)
            y_hat.extend(outputs.tolist())
            loss = loss_fn(outputs , dist_true)
            final_loss += loss.item()
    return (final_loss / len(data_loader)) , np.array(y_hat)

def get_model() -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(settings.MODEL["input_size"], settings.MODEL["l1_size"]),
        nn.BatchNorm1d(settings.MODEL["l1_size"]),
        nn.ReLU(),
        ###################
        nn.Linear(settings.MODEL["l1_size"] , settings.MODEL["l2_size"]),
        nn.BatchNorm1d(settings.MODEL["l2_size"]),
        nn.ReLU(),
        ###################
        nn.Linear(settings.MODEL["l2_size"], settings.MODEL["l3_size"]),
        nn.BatchNorm1d(settings.MODEL["l3_size"]),
        nn.ReLU(),
        ###################
        nn.Linear(settings.MODEL["l3_size"] , settings.MODEL["output_size"]),
        nn.ReLU()
    )

    return model

def get_simple_model() -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(settings.SIMPLE_MODEL["input_size"] , 400),
        nn.BatchNorm1d(400),
        nn.ReLU(),
        ###################
        nn.Linear(400 , 300),
        nn.BatchNorm1d(300),
        nn.ReLU(),
        ###################
        nn.Linear(300 , settings.SIMPLE_MODEL["output_size"]),
        nn.ReLU()
    )
    return model

def load_model() -> nn.Sequential:
    model_state = tch.load(open(settings.CHECKPOINTS["MODEL"] , "rb"))
    model = get_simple_model()
    model.load_state_dict(model_state)
    return model

def load_svm_model():
    path = "svm_model.pt"
    svm_model = pickle.loads(open(path , "rb").read())
    return svm_model

def load_embeddings() -> np.array:
    return pickle.load(open(settings.DUMP_FILES["EMB_DATA"] , "rb"))

def load_graph() -> nx.Graph:
    return nx.read_edgelist(settings.DATA_FILES["GRAPH"])

def find_path(src, dest) -> list:
    pass

def predict_distance(src: str, dst: str) -> int:
    embeddings = load_embeddings()
    src_embeddings = np.array(embeddings[int(src)])
    dst_embeddings = np.array(embeddings[int(dst)])

    print(src_embeddings)
    print(dst_embeddings)

    model_input = tch.from_numpy((src_embeddings + dst_embeddings) / 2)
    #model = load_model()
    #model.eval()
    #output = model.forward(model_input.reshape((1 , len(model_input))))
    #return output.item()

    svm_model = load_svm_model()
    return svm_model.predict(model_input.reshape(1 , len(model_input)))

def find_path(src: str , dst: str , graph: nx.Graph) -> list:
    current_node = src
    path = [current_node]
    visited_nodes = [current_node]

    while current_node != dst:
        neighbors_node = graph.neighbors(current_node)
        nodes_cost = dict()
        for node in list(neighbors_node):
            if node not in visited_nodes:
                real_dist = 0
                pred_dist = round(predict_distance(node, dst))
                nodes_cost[node] = real_dist + pred_dist

        nodes_cost = list(nodes_cost.items())
        if len(nodes_cost) == 0:
            print("Failed to predict the path")
            return 
        best_cost = min(nodes_cost,  key= lambda node : node[1])
        current_node = best_cost[0]
        visited_nodes.append(current_node)
        path.append(current_node)

    return path


def main():
    graph = load_graph()

    n_nodes = graph.number_of_nodes()
    
    src = random.randint(1 , n_nodes)
    dst = random.randint(1 , n_nodes)

    while src == dst:
        src = random.randint(1, n_nodes)
        dst = random.randint(1, n_nodes)
    
    src = str(src)
    dst = str(dst)

    # print("Path between {} and {}: ".format(src , dst))
    # print("Observed: " , nx.shortest_path(graph, src, dst))
    # print("Observed distance: " , nx.shortest_path_length(graph, src, dst))
    # print("Predicted distance: " , round(predict_distance(src , dst)))
    # print("Predicted: " , find_path(src , dst , graph))

    counter = 0
    ones_counter = 0
    acc_dict = {1 : 0 , 2 : 0 , 3 : 0}
    
    for i in range(500):
        src = random.randint(1 , n_nodes)
        dst = random.randint(1 , n_nodes)

        while src == dst:
            src = random.randint(1, n_nodes)
            dst = random.randint(1, n_nodes)

        src = str(src)
        dst = str(dst)
        
        #print(f"source= {src}, dst= {dst}")

        obs_shortest_path = nx.shortest_path_length(graph, source=src, target=dst)
        #if obs_shortest_path == 1:
            #print("One Step Distance")
            #counter += 1
            #ones_counter += 1
            #continue
        pred_pre = predict_distance(src, dst)
        pred_shortest_path = np.round(pred_pre)
        print(f"Observed= {obs_shortest_path}, Predicted= {pred_shortest_path}")
        if obs_shortest_path == pred_shortest_path:
            counter += 1
            acc_dict[obs_shortest_path] += 1
        else:
            print(f"Observed = {obs_shortest_path}, Predicted = {pred_pre}")

    print(f"ones = {round(ones_counter / 1000 , 2) * 100}%")
    print(round(counter / 500 , 2) * 100 , "%")
    print(acc_dict)
if __name__ == "__main__":
    main()

