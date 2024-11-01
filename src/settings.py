import datetime
from pathlib import Path

N_LANDMARKS = 500
GRAPH_NAME = "weighted-road-chesapeake"
#GRAPH_NAME = "road-euroroad"
DEFAULT_SEED = 9999

MODEL = {
        "learning_rate" :  0.001,
        "min_learning_rate" : 1e-05,
        "n_epochs" : 500,
        "max_learning_rate" : 0.001,
        "batch_size" : 100,
        "input_size" : 128,
        "l1_size" : 200,
        "l2_size" : 100,
        "l3_size" : 50 ,
        "output_size" : 1,
        "lr_sched" : "clr"
}

SIMPLE_MODEL = {
        "learning_rate" :  0.001,
        "min_learning_rate" : 1e-05,
        "n_epochs" : 500,
        "max_learning_rate" : 0.001,
        "batch_size" : 100,
        "input_size" : 128,
        "l1_size" : 50,
        "l2_size" : 30,
        "output_size" : 1,
        "lr_sched" : "clr"
}


ROOT_PATH = {
    "GRAPH" : Path("../graph/"),
    "EMB" : Path("../emb/"), 
    "OUTPUTS" : Path("../outputs/"),    
    "DUMP" : Path("../dump/") , 
    "MODELS" : Path("../model/"), 
    "CHECKPOINTS" : Path("../checkpoints/"),
    "IMAGES" : Path("../images/"), 
    "LOG" : Path("../log/"),
}

ENABLE_LOGGING = True

DATA_FILES = {
    "GRAPH" : Path(ROOT_PATH["GRAPH"] , GRAPH_NAME+".edgelist") , 
    "EMB" : Path(ROOT_PATH["EMB"] , GRAPH_NAME+".emb"),
}


DUMP_FILES = {
        "EMB_DATA" : Path(ROOT_PATH["DUMP"] , "{}_embeddings.pk".format(GRAPH_NAME)),
        "DIST_DATA" : Path(ROOT_PATH["DUMP"] , "{}_distances.pk".format(GRAPH_NAME)),
        "TRAIN" : Path(ROOT_PATH["DUMP"] , "{}_training_data.pk".format(GRAPH_NAME)), 
        "VALIDATE" :  Path(ROOT_PATH["DUMP"] , "{}_validation_data.pk".format(GRAPH_NAME)),
        "TEST" : Path(ROOT_PATH["DUMP"] , "{}_testing_data.pk".format(GRAPH_NAME)),
}

OUTPUT_FILES = {
        "MODEL" : Path(ROOT_PATH["OUTPUTS"] , "{}_model_{}.pt".format(GRAPH_NAME , str(datetime.datetime.now()))),
        "STATE" : Path(ROOT_PATH["OUTPUTS"] , "{}_state_{}.pt".format(GRAPH_NAME , str(datetime.datetime.now()))),
        "OPTIMIZATION" : Path(ROOT_PATH["OUTPUTS"] , "{}_optim_{}.pt".format(GRAPH_NAME, str(datetime.datetime.now()))), 
        "LR_SCHED" : Path(ROOT_PATH["OUTPUTS"] , "{}_lr_sched_{}.pt".format(GRAPH_NAME , str(datetime.datetime.now()))),
}

CHECKPOINTS = {
       "STATE" : Path(ROOT_PATH["CHECKPOINTS"], GRAPH_NAME , "{}_checkpoint_state.pt".format(GRAPH_NAME)),
       "MODEL" : Path(ROOT_PATH["CHECKPOINTS"] , GRAPH_NAME,  "{}_checkpoint_model.pt".format(GRAPH_NAME)),
       "OPTIMIZATION" : Path(ROOT_PATH["CHECKPOINTS"],GRAPH_NAME , "{}_checkpoint_optim.pt".format(GRAPH_NAME)),
       "LR_SCHED" : Path(ROOT_PATH["CHECKPOINTS"] , GRAPH_NAME,  "{}_checkpoint_sched.pt".format(GRAPH_NAME)),
        "GRAPH" : Path(ROOT_PATH["CHECKPOINTS"] , GRAPH_NAME),
}

LOG_FILES = {
    "GRAPH" : Path(ROOT_PATH["LOG"] , GRAPH_NAME),
}

