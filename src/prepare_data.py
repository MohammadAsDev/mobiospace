import pickle
import math
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from collections import Counter
import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

"""
    Responsible on getting data matrices (embeddings and distances)
    returns embeddings matrix (x) and distance matrix (y)
"""
def read_raw_data() -> tuple:
    print("Reading data matrices...")
    x = pickle.load(open(settings.DUMP_FILES["EMB_DATA"] , "rb"))
    y = pickle.load(open(settings.DUMP_FILES["DIST_DATA"] , "rb"))
    return (x, y)

def hist_plt(title : str, xlabel : str, ylabel : str, data : np.array) -> None:
    plt.figure()
    plt.title(title)
    plt.suptitle(settings.GRAPH_NAME)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    sns.histplot(data=pd.DataFrame(data)  , kde=True)
    plt.savefig(settings.ROOT_PATH["IMAGES"].joinpath("{}_{}".format(settings.GRAPH_NAME,title)), dpi=200)

def pie_plt(title : str, data : list, labels : list, colors: list):
    #plt.figure(figsize=(500, 500))
    plt.figure()
    plt.suptitle(settings.GRAPH_NAME)
    plt.title(title)
    plt.pie(data, 
            labels=labels, 
            colors=colors, 
            shadow=True)

    plt.savefig(settings.ROOT_PATH["IMAGES"].joinpath("{}_{}".format(settings.GRAPH_NAME,title)), dpi=200)



"""
    Counting distance frequences in the distance matrix (y)
    returns a dictionary that maps distance to frequences
"""
def measure_distances(distances_matrix : np.array, with_plot=True) -> dict:
    print("Counting distances...")
    distanes_counter = Counter(distances_matrix.tolist())
    if with_plot:
        hist_plt(
                title="Distances in the input matrix" , 
                xlabel="Distance" , ylabel="Feq." ,
                data=distances_matrix
        ) 
    return dict(distanes_counter)

"""
    undersample the strong values
    takes undersample dictionary and training dataset
    returns the undersampled dataset
"""
def undersample_data(undersample_dict: dict, training_emb: np.array , training_dist: np.array) -> tuple:
    under_sampler = RandomUnderSampler(sampling_strategy= undersample_dict , random_state = settings.DEFAULT_SEED)
    training_emb, training_dist = under_sampler.fit_resample(training_emb , training_dist)
    return training_emb, training_dist
"""
    oversample the weak values
    takes oversample dictionary and training dataset
    returns oversampled training dataset
"""
def oversample_data(oversample_dict: dict, training_emb : np.array, training_dist: np.array) -> tuple:
    over_sampler = RandomOverSampler(sampling_strategy = oversample_dict , random_state = settings.DEFAULT_SEED)
    training_emb , training_dist = over_sampler.fit_resample(training_emb,  training_dist)
    return training_emb, training_dist

"""
    Balance the training data 
    takes the embeddings matrix and distance matrix as arguments
    returns balanced training data
"""
def balance_data(training_emb , training_dist, with_plot=True) -> tuple:
    print("Balance Training data")
    distance_freq_pairs = list(measure_distances(training_dist).items())
    distance_freq_pairs.sort(key=lambda item : item[1] , reverse=True)
    
    undersampling_percentage = 0.1  # random value
    undersampling_limit = math.ceil(len(distance_freq_pairs) * undersampling_percentage) 

    balance_rate = 0.5  # taken from the data
    print("Max frequent distances (highest 10%): " , )

    max_freq_distances = distance_freq_pairs[:undersampling_limit]

    majority_rate = int(np.array([freq for _, freq in max_freq_distances]).mean())
    minority_rate = int(majority_rate * balance_rate)
    
    print("Maj. rate: " , majority_rate)
    print("Min. rate: " , minority_rate)

    # print("Before any balancing operations: " , np.unique(training_dist , return_counts=True))
    # undersample_dict = {dist : minority_rate for dist, _ in max_freq_distances}
    # training_emb, training_dist = undersample_data(undersample_dict, training_emb, training_dist)

    # print("After undersampling: " , np.unique(training_dist , return_counts=True))

    # if with_plot:
    #     hist_plt(
    #             title="Undersampled distances matrix" ,
    #             xlabel="Distances" ,
    #             ylabel="Feq." ,
    #             data=training_dist
    #     ) 
    
    oversample_dict = {dist : majority_rate for dist, _ in distance_freq_pairs[undersampling_limit:]}
    training_emb, training_dist = oversample_data(oversample_dict, training_emb, training_dist)

    if with_plot:
        hist_plt(
            title="Balanced distances matrix",
            xlabel="Distances",
            ylabel="Freq.",
            data= training_dist
        )
    
    print("After oversampling: " , np.unique(training_dist , return_counts=True))
    print("Shape of training data after balancing: " , training_emb.shape , training_dist.shape)
    return (training_emb , training_dist)

"""
    Shuffle the training dataset to train the model again at the same model
    takes the training data (embeddings) and target data (distances)
    returns the shuffled training dataset (shuffled embeddings and distances)
"""
def shuffle_datasets(training_data, target_data):
    assert len(training_data) == len(target_data)
    perms = np.random.permutation(len(training_data))
    return training_data[perms] , target_data[perms]

"""
    Normalize model inputs, by mapping embeddings range to be between 0 and 1
    takes the a tuple that contains embeddings of the model in all the phases
    return the normalized embeddings in the same order
"""
def normalize_inputs(inputs : tuple) -> tuple:
    std_scale = MinMaxScaler(feature_range=(0, 1))

    training_emb = std_scale.fit_transform(inputs[0])   # training inputs
    validation_emb = std_scale.transform(inputs[1])     # validation inputs
    testing_emb = std_scale.transform(inputs[2])        # tesitng inputs

    return (training_emb , validation_emb , testing_emb)

"""
    split the data to different datasets (training dataset, validation dataset, testing dataset)
    takes the inputs and the outputs of the model
    returns a dictonary that contains different datasets, in different levels
"""
def split_data(training_data, target_data) -> dict:
    print("Splitting data...")

    dist_counter = dict(Counter(target_data))
    dist_pairs = list(dist_counter.items())
    dist_pairs.sort(key=lambda pair : pair[1])
    
    dist_pairs = list(filter(lambda pair : pair[1] >= 10 , dist_pairs))
    selected_dist = list(map(lambda pair : pair[0] , dist_pairs))

    dist_ind = np.array([i for i in range(len(target_data)) if target_data[i] in selected_dist])
    target_data = target_data[dist_ind]
    training_data = training_data[dist_ind]

    hist_plt("Removing outlier distances", "distances",  "Freq.", target_data)
    

    training_emb, testing_emb , training_dist, testing_dist = train_test_split( \
            training_data , target_data , train_size=0.75 , test_size=0.25,  
            shuffle=True, random_state=settings.DEFAULT_SEED, stratify=target_data) 

    train_test_data_title = "Testing data and Training data"
    testing_training_len = [len(training_dist) + len(training_emb), len(testing_dist) + len(testing_emb)]
    
    pie_plt(data=testing_training_len , title=train_test_data_title , labels=["Training data" , "Testing data"], colors=["royalblue" , "slategrey"])

    training_emb, validation_emb , training_dist, validation_dist = train_test_split( \
            training_emb , training_dist , train_size=0.8 , test_size=0.2  
            , shuffle=True, random_state=settings.DEFAULT_SEED , stratify=training_dist)

    train_valid_data_title = "Validation data and Training data"
    validation_training_len = [len(training_dist) + len(training_emb), len(validation_emb) + len(validation_dist)]
    
    pie_plt(data=validation_training_len , title=train_valid_data_title , labels=["Training data" , "Validation data"], colors=["royalblue" , "slategrey"])

    print('shapes of train, validation, test data', training_emb.shape, training_dist.shape, validation_emb.shape, validation_dist.shape, testing_emb.shape, testing_dist.shape)

    dataset = {
        "training": (training_emb , training_dist),
        "validation" : (validation_emb , validation_dist),
        "testing" : (testing_emb , testing_dist),
    }

    return dataset


def main():
    x , y = read_raw_data() 
    dataset = split_data(x , y)
    
    training_emb, training_dist = dataset["training"]
    validation_emb , validation_dist = dataset["validation"]
    testing_emb, testing_dist = dataset["testing"]


    # Normalization
    model_inputs = (training_emb , validation_emb , testing_emb)
    training_emb, validation_emb, testing_emb = normalize_inputs(model_inputs)

    # Balance training data
    training_emb , training_dist = balance_data(training_emb , training_dist)

    # Shuffle data
    training_emb, training_dist = shuffle_datasets(training_emb , training_dist)
       

    # Saving data
    pickle.dump((training_emb , training_dist) , open(settings.DUMP_FILES["TRAIN"] , "wb"))
    pickle.dump((validation_emb , validation_dist) , open(settings.DUMP_FILES["VALIDATE"] , "wb"))
    pickle.dump((testing_emb , testing_dist) , open(settings.DUMP_FILES["TEST"] , "wb"))


if __name__ == "__main__":
    sns.set_theme()
    main()

