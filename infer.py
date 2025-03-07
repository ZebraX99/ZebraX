import os
import sys
import csv
import json
import joblib
import logging
import numpy as np
import pandas as pd
from pprint import pprint
import trajectorytools as tt
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from train import generate_classifer_data, reorganize_data_for_classifer
import argparse
from idtrackerai.video.general_video import generate_trajectories_video
from idtrackerai import Session

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # 将日志输出到标准输出（stdout）
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train classifier for fish health')
    parser.add_argument('data_path', type=str, help='Path to processed videos', default='processed_videos/test/session_VID_20250207_050_e')
    return parser.parse_args()

def load_models(output_folder, n_splits):
    models = []
    for i in range(n_splits):
        model_path = os.path.join(output_folder, f'model_fold_{i+1}.joblib')
        model = joblib.load(model_path)
        models.append(model)
        # print(f"Model loaded from: {model_path}")
    return models

def model_infer(models, x_val, ori_ids):

    y_preds = []
    for i, clf in enumerate(models):
        y_pred = clf.predict(x_val)
        y_preds.append(y_pred)

    # Utilize the mode to vote for final prediction
    y_preds = np.array(y_preds).T
    final_predictions = []
    for preds in y_preds:
        final_prediction = np.bincount(preds).argmax()
        final_predictions.append(final_prediction)

    # assign the original fish id to the final prediction
    final_predictions = np.array(final_predictions)
    unique_idx = np.unique(ori_ids)
    final_id_predictions = {i : [] for i in unique_idx}
    for id in unique_idx:
        idx = np.where(ori_ids == id)
        final_id_predictions[id] = final_predictions[idx]

    # vote for final prediction
    vote_predictions = []
    vote_prob = []
    for id in unique_idx:
        # utilize the mode to define final prediction for each id
        vote_predictions.append(np.bincount(final_id_predictions[id]).argmax())
        # calc the probability of the final prediction
        vote_prob.append(np.bincount(final_id_predictions[id]) / len(final_id_predictions[id]))


    # define the degree of diagnoisis
    healthy_degree = [1.0, 0.85, 0.5]
    unhealthy_degree = [1.0, 0.85, 0.5]

    logging.info('Final prediction:')
    pred_results = []
    for id in unique_idx:
        prob = vote_prob[id]
        pred = vote_predictions[id]
        prob = prob[pred]
        if pred == 0:
            if prob <= healthy_degree[0] and prob > healthy_degree[1]:
                logging.info(f'Fish {id} is predicted as healthy')
                pred_results.append('healthy')
            if prob <= healthy_degree[1] and prob > healthy_degree[2]:
                logging.info(f'Fish {id} is predicted as healthy but with potential risk')
                pred_results.append('healthy with potential risk')
        if pred == 1:
            if prob <= unhealthy_degree[0] and prob > unhealthy_degree[1]:
                logging.info(f'Fish {id} is predicted as diabetic')
                pred_results.append('diabetic')
            if prob <= unhealthy_degree[1] and prob > unhealthy_degree[2]:
                logging.info(f'Fish {id} is predicted as mild diabetic')
                pred_results.append('mild diabetic')

    return pred_results

if __name__ == '__main__':
    # Load models
    models_path = 'output_models'
    n_splits = 5
    models = load_models(models_path, n_splits)

    args = parse_args()
    # Load data
    data_path = args.data_path
    data_path = os.path.join(data_path, 'trajectories/without_gaps.npy')

    output_folder = 'infer_classifier_data'
    generate_classifer_data(data_path, output_folder, label=2, infer=True)

    # Reorganize data
    X, Y, ori_ids = reorganize_data_for_classifer(os.path.join(output_folder, 'classifier_data_2.csv'), infer=True)

    # Inference
    pred_results = model_infer(models, X, ori_ids)

    # save in the session.json
    session_path = data_path.replace('trajectories/without_gaps.npy', 'session.json')
    with open(session_path, 'r') as f:
        session = json.load(f)
    session['diabetes_predictions'] = pred_results
    with open(session_path, 'w') as f:
        json.dump(session, f)

    session = Session.load(session_path)
    # Generate video
    possible_files = (
        "validated.npy",
        "without_gaps.npy",
        "with_gaps.npy",
        "trajectories_validated.npy",
        "trajectories_wo_gaps.npy",
        "trajectories.npy",
        "trajectories_wo_identification.npy",
    )
    for file in possible_files:
        path = session.trajectories_folder / file
        if path.is_file():
            logging.info("Loading trajectories from %s", path)
            trajectories = np.load(path, allow_pickle=True).item()["trajectories"]
            break
    generate_trajectories_video(
        session,
        trajectories,
        draw_in_gray=False,
        centroid_trace_length=30,
        starting_frame=100,
        ending_frame=1000,
    )