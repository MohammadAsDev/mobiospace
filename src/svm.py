from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import settings

import random


def load_training_data() -> tuple:
    return pickle.load(open(settings.DUMP_FILES["TRAIN"] , "rb"))

def load_testing_data():
    return pickle.load(open(settings.DUMP_FILES["TEST"] , "rb"))

"""
    Shuffle datasets before use it
    takes training dataset and testing dataset
    returns the shuffled version of training and testing dataset
"""
def shuffle_data(training_data, testing_data):
    shuffle_perms = np.random.permutation(len(training_data))
    return training_data[shuffle_perms],  testing_data[shuffle_perms]

"""
    Returns the distance prediction accuracy
    takes the observed distance and the predicted distance
    returns a dictionary that maps each distance to its prediction accuracy
"""
def get_dist_accuracy(obs_dist : np.array, pred_dist : np.array) -> dict:
    dist_indexes = dict()
    correct_predictions = dict()
    dist_acc = dict()
    
    for distance in np.unique(pred_dist):
        indecies = np.where(obs_dist == distance)[0]
        if len(indecies) > 0:
            dist_indexes[distance] = indecies

    for distance, indecies in dist_indexes.items():
        correct_predictions[distance] = len(np.where(pred_dist[indecies] == distance)[0])

    for distance in dist_indexes.keys():
        dist_acc[distance] = round(correct_predictions[distance] / len(dist_indexes[distance]) , 2) * 100

    return dist_acc

def plot_model_accuracy(y_test : np.array, y_class : np.array) -> None:
    dist_acc = get_dist_accuracy(y_test, y_class)
    print("Prediction Accuracy for Each Distance:")
    print(dist_acc)

    accuracy_labels = {dist : "Acc. {}%".format(dist_acc[dist]) for dist in dist_acc.keys()} # if you want to add some labels
    
    plt.figure()
    plt.title("SVM Accuracy")
    plt.suptitle(settings.GRAPH_NAME)
    plt.xlabel("disatnce")
    plt.ylabel("Acc.")
    
    acc_list = np.array(list(dist_acc.values()))
    dist_list = np.array(list(dist_acc.keys()))
    
    acc_dist_frame = pd.DataFrame(data=np.array([acc_list]) , columns=dist_list)
    
    bars = sns.barplot(data=acc_dist_frame)
    
    if len(dist_list) > 10:
        bars.set(xticklabels=[])

    plt.savefig(settings.ROOT_PATH["IMAGES"].joinpath("{}_{}".format(settings.GRAPH_NAME , "svm_acc")) , dpi=200)



def main():
    x_train, y_train  = load_training_data()
    x_train, y_train = shuffle_data(x_train , y_train)
    svm_model = SVC(kernel="poly").fit(x_train,  y_train)

    x_test, y_test = load_testing_data()

    y_pred = svm_model.predict(x_test)
    y_class = np.round(y_pred)

    #plot_model_accuracy(y_test , y_class)

    baseline_acc = accuracy_score(y_test, y_class)*100
    baseline_mse = mean_squared_error(y_test, y_pred)
    baseline_mae = mean_absolute_error(y_test, y_pred)

    print("Baseline: Accuracy={}%, MSE={}, MAE={}".format(round(baseline_acc, 2), round(baseline_mse,2), round(baseline_mae,2)))
    print("Classification Report:")
    print(classification_report(y_test,  y_class))

    pickle.dump(svm_model , open("main_svm_model.pt" , "wb"))

    cm = confusion_matrix(y_test , y_class)
    sns.heatmap(cm  , fmt='g', annot=True , xticklabels=np.unique(y_test) , yticklabels=np.unique(y_test) , cmap="Blues")

    plt.ylabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17, pad=20)
    plt.gca().xaxis.set_label_position('top') 
    plt.xlabel('Prediction', fontsize=13) 
    plt.gca().xaxis.tick_top() 
    plt.gca().figure.subplots_adjust(bottom=0.2) 
    #plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
    #plt.show()

    plt.savefig(settings.ROOT_PATH["IMAGES"].joinpath("{}_{}".format(settings.GRAPH_NAME , "svm_conf")) , dpi=200)


    """
    

    acc = 0.0
    emb_lines = open("../emb/weighted-road-chesapeake.emb").readlines()[1:]
    emb_dict = {}
    for line in emb_lines:
        temp = line.split(" ")
        emb_dict[int(temp[0])] = np.array(temp[1:] , dtype=np.float32)
  


    scaler = MinMaxScaler(feature_range=(0,1))
    
    embs = []
    
    for emb in emb_dict.values():
        embs.append(emb)
        
    print(np.array(embs))
    
    scaler.fit(np.array(embs))
    
    for key, val in emb_dict.items():
        scaled_emb = scaler.transform(val.reshape(1 , -1))
        emb_dict[key] = scaled_emb.reshape(1 , -1)
    
    for i in range(1000):
        start_n = np.random.randint(1 , 40)
        end_n = np.random.randint(1 , 40)
        if start_n == end_n:
            i -= 1
            continue
        s = emb_dict[start_n]
        d = emb_dict[end_n]
        
        m = np.array((s + d) / 2).reshape(1 , -1)
        
        pred_dist = svm_model.predict(m.reshape(1 , -1))
        print(pred_dist)
    """

    #acc = 0.0

    #for i in range(1000):
        #sample_i = random.randint(1 , len(x_test)) - 1
        #input_emb = x_test[sample_i]
        #obs_dist = y_test[sample_i]
        #pred_dist = svm_model.predict(input_emb.reshape(1 , -1))
        #if obs_dist == pred_dist:
            #acc += 1
            
    #acc /= 1000
    
    #print("Accuracy: " , acc)


if __name__ == "__main__":
    sns.set_theme()
    main() 
