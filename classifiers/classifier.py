import pandas as pd
import lightgbm as lgb
import joblib
import numpy as np
import os

# Get the directory where classifier.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to lgb.pkl
classifier_path = os.path.join(script_dir, "lgb.pkl")

classifier = joblib.load(classifier_path)


def classify(data: pd.DataFrame):
    """
    Performs state classification using I1Q1,I2Q2 data.

    Returns labels
    0: ground
    1: excited
    2: f
    """
    try:
        data = data.drop(columns="Unnamed: 0")
    except:
        pass

    pred = classifier.predict(data)
    data["predicted"] = pred

    return data


def reshape_for_exp(data: pd.Series, reps: int, num_steps: int):
    """This takes in the list of I or Q data and rearranges it such that we
    have num steps columns and reps rows."""
    # first input is number of rows
    # second input is number of columns
    total_data_size = reps * num_steps
    data_cut = data[0:total_data_size]
    arr = data_cut.values
    new_arr = np.reshape(arr, (reps, -1))

    return new_arr


def probabilities(arr: np.array):
    """Returns probabilities for each state (0, 1, 2) as P_g, P_e, and P_f
    respectively."""
    # THIS TAKES AN ARRAY IN THE SHAPE (REPS, STEPS) IT AVERAGES COLUMN WISE TO AVERAGE PER STEP
    prob = [np.mean(arr == i, axis=0) for i in range(3)]

    P_g = prob[0]
    P_e = prob[1]
    P_f = prob[2]

    prob_dict = {"P_g": P_g, "P_e": P_e, "P_f": P_f}
    return prob_dict


def population(arr: np.array):
    """Returns populations for each state (0, 1, 2)."""
    # THIS TAKES AN ARRAY IN THE SHAPE (REPS, STEPS) IT AVERAGES COLUMN WISE TO AVERAGE PER STEP
    prob = [np.sum(arr == i, axis=0) for i in range(3)]

    Pop_g = prob[0]
    Pop_e = prob[1]
    Pop_f = prob[2]

    pop_dict = {"Pop_g": Pop_g, "Pop_e": Pop_e, "Pop_f": Pop_f}
    return pop_dict
