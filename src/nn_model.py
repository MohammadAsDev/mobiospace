import torch as tch 
import torch.nn as nn
from torch.optim import SGD , RMSprop
from torch.utils import data as torch_data
import networkx as nx 
import random

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pickle

import settings

import copy
import os


def load_embeddings() -> np.array:
    return pickle.load(open(settings.DUMP_FILES["EMB_DATA"] , "rb"))

def load_graph() -> nx.Graph:
    return nx.read_weighted_edgelist(settings.DATA_FILES["GRAPH"])

def load_training_data():
    return pickle.load(open(settings.DUMP_FILES["TRAIN"] , "rb"))

def load_validation_data():
    return pickle.load(open(settings.DUMP_FILES["VALIDATE"] , "rb"))

def load_testing_data():
    return pickle.load(open(settings.DUMP_FILES["TEST"] , "rb"))

"""
    Make the neural network
    takes nothing
    returns the model
"""
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
        nn.Linear(settings.SIMPLE_MODEL["input_size"] , settings.SIMPLE_MODEL["l1_size"]),
        nn.BatchNorm1d(settings.SIMPLE_MODEL["l1_size"]),
        nn.ReLU(),
        ###################
        nn.Linear(settings.SIMPLE_MODEL["l1_size"] , settings.SIMPLE_MODEL["l2_size"]),
        nn.BatchNorm1d(settings.SIMPLE_MODEL["l2_size"]),
        nn.ReLU(),
        ###################
        nn.Linear(settings.SIMPLE_MODEL["l2_size"] , settings.SIMPLE_MODEL["output_size"]),
        nn.ReLU()
    )
    return model

"""
    Calculate the poisson loss
    takes the predicted and observed outputs
    return the final loss
"""
def poisson_loss(y_pred , y_true):
    y_pred = tch.squeeze(y_pred)
    loss = tch.mean(y_pred - y_true * tch.log(y_pred+1e-7))
    return loss

"""
    Evaluate the model
    takes the model, the loss function, and the dataset loader
    returns the mean loss
"""
def evaluate(model , loss_fn , data_loader):
    model.eval()
    final_loss = 0.0
    count = 0
    with tch.no_grad():
        for data in data_loader:
            inputs, dist_true = data[0] , data[1]
            count += len(data)
            outputs = model(inputs)
            loss = loss_fn(outputs, dist_true)
            final_loss += loss.item()
    return final_loss / len(data_loader)

"""
    Test the model
    takes the model, the loss function, and the dataset loader
    returns the mean loss and the predicted output
"""
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

"""
    Initialize the dataloader for each tensor dataset
    takes the tensor dataset content (in this case the embeddings and the distance)
    returns the dataloader object
"""
def init_data_loader(embeddings , dist):
    data_set = torch_data.TensorDataset(tch.as_tensor(embeddings , dtype=tch.float32) , tch.as_tensor(dist , dtype=tch.float32))
    data_loader = torch_data.DataLoader(data_set , settings.MODEL["batch_size"] , drop_last=False)
    return data_loader

"""
    Save the model's current state
    takes the model's state
    retuns nothing
"""
def save_checkpoint(state):
    if not os.path.exists(settings.ROOT_PATH["CHECKPOINTS"]):
        os.mkdir(settings.ROOT_PATH["CHECKPOINTS"])
    tch.save(state, settings.CHECKPOINTS["STATE"])

"""
   Inialize the tensorboard writer
"""
def get_writer():
    if not os.path.exists(settings.LOG_FILES["GRAPH"]):
        os.mkdir(settings.LOG_FILES["GRAPH"])
    return SummaryWriter(settings.LOG_FILES["GRAPH"])

"""
    Calculate the prediction accuracy for each distance
    takes the predicted and observed distance
    returns a dictiony that maps each distance to its accuracy
"""
def get_dist_accuracy(pred_dist, obs_dist):
    dist_indx = dict()
    dist_acc = dict()
    obs_dist = obs_dist[:len(pred_dist)]

    for distance in np.unique(obs_dist):
        indexes = np.where(obs_dist == distance)[0]
        if len(indexes) > 0:
            dist_indx[distance] = indexes

    pred_dist = np.round(pred_dist)
    for distance, indexes in dist_indx.items():
        n_correct_pred = len(np.where(distance == pred_dist[indexes])[0])
        dist_acc[distance] = round(n_correct_pred / len(indexes) , 2) * 100

    return dist_acc



def plt_dist_accuracy(pred_dist : np.array , obs_dist : np.array , currnet_run : int):
    dist_acc = get_dist_accuracy(pred_dist , obs_dist)
    print("Prediction Accuracy for Each Distance:")
    print(dist_acc)

    accuracy_labels = {dist : "Acc. {}%".format(dist_acc[dist]) for dist in dist_acc.keys()} # if you want to add some labels
    
    plt.figure()
    plt.title("Neural Network Accuracy")
    plt.suptitle(settings.GRAPH_NAME)
    plt.xlabel("disatnce")
    plt.ylabel("Acc.")
    
    acc_list = np.array(list(dist_acc.values()))
    dist_list = np.array(list(dist_acc.keys()))
    
    acc_dist_frame = pd.DataFrame(data=np.array([acc_list]) , columns=dist_list)
    
    bars = sns.barplot(data=acc_dist_frame)
    
    if len(dist_list) > 10:
        bars.set(xticklabels=[])

    plt.savefig(settings.ROOT_PATH["IMAGES"].joinpath("{}_{}_{}".format(settings.GRAPH_NAME , "nn_acc" , currnet_run)) , dpi=200)

def predict_distance( model: nn.Sequential, src: str, dst: str) -> int:
    embeddings = load_embeddings()
    src_embeddings = np.array(embeddings[int(src) - 1])
    dst_embeddings = np.array(embeddings[int(dst) - 1])

    model_input = tch.from_numpy((src_embeddings + dst_embeddings) / 2)
    model.eval()
    output = model.forward(model_input.reshape((1 , len(model_input))))
    return output.item()

def real_testing(model):
    graph = load_graph()
    n_nodes = graph.number_of_nodes()
    counter = 0
    for i in range(500):
        src = random.randint(1 , n_nodes)
        dst = random.randint(1 , n_nodes)

        while src == dst:
            src = random.randint(1, n_nodes)
            dst = random.randint(1, n_nodes)

        src = str(src)
        dst = str(dst)

        obs_shortest_path = nx.shortest_path_length(graph, source=src, target=dst)
        pred_shortest_path = np.round(predict_distance(model , src, dst))
        if obs_shortest_path == pred_shortest_path:
            counter += 1

    print(round(counter / 500 , 2) * 100 , "%")

def main():
    min_val_loss = np.inf
    early_stop_patience = 50

    writer = get_writer()

    model = get_simple_model()
    best_model = None
    loss_fn = poisson_loss

    training_emb, training_dist = load_training_data()
    validation_emb, validation_dist = load_validation_data()
    testing_emb, testing_dist = load_testing_data()
    

    writer.add_embedding(training_emb , tag="training embeddings")
    writer.add_embedding(validation_emb, tag="validation embeddings")
    writer.add_embedding(testing_emb, tag="testing embeddings")
    writer.flush()

    
    training_dataLoader = init_data_loader(training_emb , training_dist)
    validation_dataLoader = init_data_loader(validation_emb , validation_dist)
    testing_dataLoader = init_data_loader(testing_emb , testing_dist)


    optimizer = RMSprop(
            model.parameters() , 
            lr=settings.MODEL["learning_rate"] , 
            alpha=0.99 , 
            eps=1e-08 , 
            weight_decay = 0 , 
            momentum = 0 , 
            centered=False
        )
    lr_sched = tch.optim.lr_scheduler.CyclicLR(
            optimizer , 
            settings.MODEL["min_learning_rate"] , 
            settings.MODEL["max_learning_rate"] , 
            mode='triangular'  , 
            last_epoch = -1  , 
            step_size_down= None , 
            step_size_up = 8 * len(training_dataLoader), 
            gamma=0.95
        )

    
    starting_epoch = 0
    patience_counter = 0
    n_runs = 1


    if os.path.exists(settings.CHECKPOINTS["GRAPH"]):   # loads the previous the state of the model
        model.load_state_dict(tch.load(settings.CHECKPOINTS["MODEL"]))
        optimizer.load_state_dict(tch.load(settings.CHECKPOINTS["OPTIMIZATION"]))
        lr_sched.load_state_dict(tch.load(settings.CHECKPOINTS["LR_SCHED"]))
        state = tch.load(settings.CHECKPOINTS["STATE"])

        starting_epoch = state['epoch']
        n_runs = state["last_run"] + 1

    if starting_epoch >= settings.MODEL["n_epochs"] - 1:    # training is ended (no more epochs)
        print("Training is already done!")
        return 

    writer.add_graph(model, tch.from_numpy(training_emb))
    writer.flush()

    for epoch in range(starting_epoch , settings.MODEL["n_epochs"]):    # the training loop
        running_loss = 0.0

        for i , data in enumerate(training_dataLoader): # training the model
            model.train()
            inputs, dist_true = data[0] , data[1]
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, dist_true)
            writer.add_scalar("model loss" , loss.item() , i)


            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            lr_sched.step()

        val_loss = evaluate(model, loss_fn, validation_dataLoader)  # evaluate the model

        if val_loss < min_val_loss: # save the best model
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
            print("Best loss value= {}, model is saved".format(min_val_loss))

        else:
            patience_counter += 1

        if patience_counter > early_stop_patience:  # early stopping the process 
            print("Early stopping at epoch={}, with current loss={}".format(epoch , loss))
            break

        if epoch % 10 == 0: # save a checkpoint
            print("Saving a checkpoint (epoch = {}, min_loss = {})".format(epoch , min_val_loss))
            if not os.path.exists(settings.CHECKPOINTS["GRAPH"]):
                os.mkdir(settings.CHECKPOINTS["GRAPH"])

            tch.save(best_model.state_dict(), settings.CHECKPOINTS["MODEL"])
            tch.save(optimizer.state_dict(), settings.CHECKPOINTS["OPTIMIZATION"])
            tch.save(lr_sched.state_dict(), settings.CHECKPOINTS["LR_SCHED"])


        state = {   # update model's current state
            "epoch" : epoch + 1,
            "model_state" : model.state_dict(),
            "optim_state" : optimizer.state_dict(),
            "last_run" : n_runs
        }        

    test_loss, y_hat = test(model, loss_fn, testing_dataLoader) # testing the model
    print("Testing Loss: " , test_loss) 
    print("Model Accuracy: {}%".format( str ( round( accuracy_score(testing_dist[:len(y_hat)] , np.round(y_hat)) * 100 , 2)) ))
    print("Saving the model...")

    if not os.path.exists(settings.ROOT_PATH["OUTPUTS"]):
        os.mkdir(settings.ROOT_PATH["OUTPUTS"])

    # save the model's final state
    save_checkpoint(state)
    tch.save(best_model.state_dict() , settings.CHECKPOINTS["MODEL"])
    tch.save(optimizer.state_dict() , settings.CHECKPOINTS["OPTIMIZATION"])
    tch.save(lr_sched.state_dict() , settings.CHECKPOINTS["LR_SCHED"])

    real_testing(best_model)

    plt_dist_accuracy(y_hat, testing_dist, n_runs)

if __name__ == "__main__":
    tch.manual_seed(settings.DEFAULT_SEED)
    sns.set_theme()
    main()


