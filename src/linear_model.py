from sklearn.metrics import accuracy_score , mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

import pickle

import seaborn as sns
import matplotlib.pyplot as plt

import settings

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
    plt.title("Linear Regiression Accuracy")
    plt.suptitle(settings.GRAPH_NAME)
    plt.xlabel("disatnce")
    plt.ylabel("Acc.")
    
    acc_list = np.array(list(dist_acc.values()))
    dist_list = np.array(list(dist_acc.keys()))
    
    acc_dist_frame = pd.DataFrame(data=np.array([acc_list]) , columns=dist_list)
    
    bars = sns.barplot(data=acc_dist_frame)
    
    if len(dist_list) > 10:
        bars.set(xticklabels=[])

    plt.savefig(settings.ROOT_PATH["IMAGES"].joinpath("{}_{}".format(settings.GRAPH_NAME , "linear_reg_acc")) , dpi=200)



def main():
    x_train, y_train  = load_training_data()
    x_train, y_train = shuffle_data(x_train , y_train)
    baseline_model = LinearRegression(fit_intercept=True, n_jobs=-1).fit(x_train, y_train)

    x_test, y_test = load_testing_data()

    y_pred = baseline_model.predict(x_test)
    y_class = np.round(y_pred)

    plot_model_accuracy(y_test , y_class)

    baseline_acc = accuracy_score(y_test, y_class)*100
    baseline_mse = mean_squared_error(y_test, y_pred)
    baseline_mae = mean_absolute_error(y_test, y_pred)
    print("Baseline: Accuracy={}%, MSE={}, MAE={}".format(round(baseline_acc, 2), round(baseline_mse,2), round(baseline_mae,2)))

if __name__ == "__main__":
    sns.set_theme()
    main() 
