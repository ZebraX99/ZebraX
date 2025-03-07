import os
import csv
import joblib
import numpy as np
import pandas as pd
from pprint import pprint
import trajectorytools as tt
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy import interpolate
import argparse


def remove_outliers_with_z_score(data):
    mean = np.nanmean(data)
    std_dev = np.nanstd(data)
    z_scores = (data - mean) / std_dev

    valid_index = np.abs(z_scores) < 3
    invalid_index = np.abs(z_scores) >= 3

    filter_data = data[valid_index]
    if not np.all(valid_index):
        interp_func = interpolate.interp1d(np.where(valid_index)[0], filter_data, kind='linear', fill_value='extrapolate')
        data = interp_func(np.arange(len(data)))
    return data

def remove_outliers_with_iqr(data):
    Q1 = np.nanpercentile(data, 15)
    Q3 = np.nanpercentile(data, 85)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    valid_index = (data >= lower_bound) & (data <= upper_bound)
    filter_data = data[valid_index]

    # interpolate
    if not np.all(valid_index):
        interp_func = interpolate.interp1d(np.where(valid_index)[0], filter_data, kind='linear', fill_value='extrapolate')
        data = interp_func(np.arange(len(data)))

    return data

def generate_classifer_data(trajectories_path, output_folder, data_frame_interval=200, label=0, infer=False):
    """
    Generate classifier data from video clips
    :param data_path: Path to video clips
    :param output_folder: Path to save classifier data
    :return:
    """

    # Check output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # check if the classifier data has been generated
    data = {
        'fish_id': [],
        'speed': [],
        'acceleration': [],
        'distance': [],
        'total_distance': [],
        'burst_distance': [],
        'total_kinetic_energy': []
    }
    data = pd.DataFrame(data)
    if os.path.exists(os.path.join(output_folder, 'classifier_data_{}.csv'.format(label))):
        if infer:
            # remove the old data
            os.remove(os.path.join(output_folder, 'classifier_data_{}.csv'.format(label)))
        else:
            # Read data
            data = pd.read_csv(os.path.join(output_folder, 'classifier_data_{}.csv'.format(label)))

    # Load trajectories
    tr = tt.Trajectories.from_idtrackerai(
        trajectories_path, interpolate_nans=True, smooth_params={'sigma': 1}
    )

    # remove first 100 frame~about 2s and last 100 frame data for stable data analysis
    tr = tr[100:]
    tr = tr[:-100]

    # Since the arena of the setup was circular and the fish visited the borders of the arena
    # we use the estimate_center_and_radius_from_locations to center the trajectories
    # in the arena
    center, radius = tr.estimate_center_and_radius_from_locations(in_px=True)
    tr.origin_to(center)

    # In our case we know that the body_length_px is a good estimate for the body length
    # since we loaded the trajectories with the method from_idtrackerai this value is
    # stored in the tr.params disctionary
    tr.new_length_unit(tr.params["body_length_px"], "BL")

    # Since we loaded the trajectories with the method from_idtrackerai we can
    # use the frame_rate variable stored in the tr.params disctioanry to
    # to set the time units to seconds
    tr.new_time_unit(tr.params["frame_rate"], "s")
    fish_number = tr.number_of_individuals
    s = tr.s
    a = tr.a
    v = tr.v
    speed = tr.speed
    acceleration = tr.acceleration
    curvature = tr.curvature

    for i in range(fish_number):
        s[:, i, 0] = remove_outliers_with_z_score(tr.s[:, i, 0])
        s[:, i, 1] = remove_outliers_with_z_score(tr.s[:, i, 1])
        a[:, i, 0] = remove_outliers_with_iqr(tr.a[:, i, 0])
        a[:, i, 1] = remove_outliers_with_iqr(tr.a[:, i, 1])
        v[:, i, 0] = remove_outliers_with_z_score(tr.v[:, i, 0])
        v[:, i, 1] = remove_outliers_with_z_score(tr.v[:, i, 1])

        speed[:, i] = remove_outliers_with_iqr(speed[:, i])
        acceleration[:, i] = remove_outliers_with_iqr(acceleration[:, i])
        curvature[:, i] = remove_outliers_with_iqr(curvature[:, i])


    # Process data to generate classifier data
    time_frame_number = tr.s.shape[0]

    # Generate classifier data
    # save in csv file format
    index = 0 if len(data) == 0 else data['fish_id'].max() + 1
    for i in range(0, time_frame_number, data_frame_interval):
        if i + data_frame_interval >= time_frame_number:
            break

        # Load data
        velocities = v[i:i+data_frame_interval]      # [t, n, 2]
        accelerations = a[i:i+data_frame_interval]   # [t, n, 2]
        distance_to_center = tr.distance_to_origin[i:i+data_frame_interval]  # [t, n]
        curva = curvature[i:i+data_frame_interval]  # [t, n]

        total_distance = np.sum(speed[i:i+data_frame_interval], axis=0) * 1 / tr.params["frame_rate"]
        burst_threshold = 2.0
        burst_distance = []
        for j in range(fish_number):
            burst_distance.append(np.sum(speed[speed[:, j] > burst_threshold, j]) * 1 / tr.params["frame_rate"])
        burst_distance = np.array(burst_distance)
        fish_mass = 2.0
        total_kinetic_energy = fish_mass * np.sum(speed[i:i+data_frame_interval] * acceleration[i:i+data_frame_interval], axis=0) * 1 / tr.params["frame_rate"]

        # transfer distance_to_center to cumulative distance
        distance_to_center_res = np.abs(distance_to_center[1:] - distance_to_center[:-1])
        distance_to_center_cum = np.cumsum(distance_to_center_res, axis=0)
        # append 0 to the first row
        distance_to_center_cum = np.insert(distance_to_center_cum, 0, 0, axis=0)

        # transfer (vx, vy) and (ax, ay)  to v and a
        speed = np.linalg.norm(velocities, axis=-1)
        acceleration = np.linalg.norm(accelerations, axis=-1)

        # organize data
        for j in range(fish_number):
            fish_data = {
                'fish_id': j + index * fish_number,
                'ori_fish_id': j,
                'speed': speed[:, j],
                'acceleration': acceleration[:, j],
                'distance': distance_to_center_cum[:, j],
                'curvature': curva[:, j],
                'label': label,
                'total_distance': total_distance[j],
                'burst_distance': burst_distance[j],
                'total_kinetic_energy': total_kinetic_energy[j]
            }
            if len(data) == 0:
                data = pd.DataFrame(fish_data)
            else:
                data = pd.concat([data, pd.DataFrame(fish_data)], axis=0)

        index = index + 1

    # save data
    save_path = os.path.join(output_folder, 'classifier_data_{}.csv'.format(label))
    data.to_csv(save_path, index=False)
    print('{} Data saved in:'.format(trajectories_path), save_path)
    # print('Data saved in:', save_path)

def read_classifier_data(data_path):
    data = pd.read_csv(data_path)

    # plot fish_id: 100 speed, acceleration, distance
    fish_id = 30
    fish_data = data[data['fish_id'] == fish_id]
    speed = fish_data['speed']
    acceleration = fish_data['acceleration']
    distance = fish_data['distance']

    speed.plot()
    plt.show()
    acceleration.plot()
    plt.show()
    distance.plot()
    plt.show()

def reorganize_data_for_classifer(classifier_data_healthy_path, classifier_data_unhealthy_path=None, infer=False):
    # Read data
    classifier_data_healthy = pd.read_csv(classifier_data_healthy_path)
    if classifier_data_unhealthy_path is not None:
        classifier_data_unhealthy = pd.read_csv(classifier_data_unhealthy_path)

    healthy_data_number = classifier_data_healthy['fish_id'].max() + 1
    if classifier_data_unhealthy_path is not None:
        unhealthy_data_number = classifier_data_unhealthy['fish_id'].max() + 1
        print('Healthy Data Number:', healthy_data_number)
        print('Unhealthy Data Number:', unhealthy_data_number)
        # change the fish_id of unhealthy data
        classifier_data_unhealthy['fish_id'] = classifier_data_unhealthy['fish_id'] + healthy_data_number

        # concat data and reorganize the fish_id
        classifier_data = pd.concat([classifier_data_healthy, classifier_data_unhealthy], axis=0)
    else:
        classifier_data = classifier_data_healthy

    grouped = classifier_data.groupby('fish_id')
    samples = []
    for fish_id, group in grouped:
        sample = {
            'fish_id': fish_id,
            'speed': group['speed'].values,
            'acceleration': group['acceleration'].values,
            'distance': group['distance'].values,
            'curvature': group['curvature'].values,
            'total_distance': group['total_distance'].values[0],
            'burst_distance': group['burst_distance'].values[0],
            'total_kinetic_energy': group['total_kinetic_energy'].values[0],
            'label': group['label'].values[0],  # assume the label is the same
            'ori_fish_id': group['ori_fish_id'].values[0]
        }
        samples.append(sample)

    X = []
    Y = []
    ori_ids = []
    for sample in samples:
        features = np.hstack((
            sample['speed'],
            sample['acceleration'],
            sample['distance'],
            sample['curvature'],
            sample['total_distance'],
            sample['burst_distance'],
            sample['total_kinetic_energy']
        ))
        X.append(features)
        Y.append(sample['label'])
        ori_ids.append(sample['ori_fish_id'])
    X = np.array(X)
    Y = np.array(Y)
    ori_ids = np.array(ori_ids)

    # select training data and testing data
    test_ratio = 0.1
    test_index = np.random.randint(0, len(X), int(len(X) * test_ratio))
    X_test = X[test_index]
    Y_test = Y[test_index]
    X_train = np.delete(X, test_index, axis=0)
    Y_train = np.delete(Y, test_index, axis=0)

    if not infer:
        return X_train, Y_train, X_test, Y_test
    else:
        return X, Y, ori_ids

def train_v1(X, Y):
    # Split data
    n_splits = 5 # default use 5 fold cross validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_models = None
    best_n_estimators = None
    best_accuracy = 0
    mean_accuracy = 0
    for n_estimators in range(10, 200, 40):
        fold = 1
        models = []
        print('Setting n_estimators:', n_estimators)
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = Y[train_index], Y[val_index]

            # Initialize classifier
            clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

            # Train classifier
            clf.fit(X_train, y_train)
            models.append(clf)

            # Predict validation set
            y_pred = clf.predict(X_val)

            # Evaluate model
            accuracy = accuracy_score(y_val, y_pred)
            report = classification_report(y_val, y_pred)

            mean_accuracy += accuracy

            print(f"Fold {fold} - 验证集准确率: {accuracy}")
            print(f"Fold {fold} - 分类报告:\n{report}")
            fold += 1

        mean_accuracy /= n_splits
        print(f"Mean Accuracy: {mean_accuracy}")
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_n_estimators = n_estimators
            best_models = models
        mean_accuracy = 0

    print('Training Done!')
    print('Best n_estimators:', best_n_estimators)
    print('Best Accuracy:', best_accuracy)

    return best_models

def train_v2(X, Y):
    # Split data
    n_splits = 5 # default use 5 fold cross validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_models = None
    best_n_estimators = None
    best_accuracy = 0
    mean_accuracy = 0
    fold = 1
    models = []

    n_estimators = 170
    print('Setting n_estimators:', n_estimators)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        # initialize classifier
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        # train classifier
        clf.fit(X_train, y_train)
        models.append(clf)

        # Predict validation set
        y_pred = clf.predict(X_val)

        # Evaluate model
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)

        mean_accuracy += accuracy

        print(f"Fold {fold} - Eval set: {accuracy}")
        print(f"Fold {fold} - Cls Report:\n{report}")
        fold += 1

    mean_accuracy /= n_splits
    print(f"Mean Accuracy: {mean_accuracy}")
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_n_estimators = n_estimators
        best_models = models
    mean_accuracy = 0

    print('Training Done!')
    print('Best n_estimators:', best_n_estimators)
    print('Best Accuracy:', best_accuracy)

    return best_models


def eval_models(models, x_val, y_val):

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

    accuracy = accuracy_score(y_val, final_predictions)
    report = classification_report(y_val, final_predictions)

    print("Final Test Accuary:", accuracy)
    print("Final Classifer Report:\n", report)


def save_models(models, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, model in enumerate(models):
        model_path = os.path.join(output_folder, f'model_fold_{i+1}.joblib')
        joblib.dump(model, model_path)
        print(f"Model saved at: {model_path}")

if __name__ == '__main__':
    videos_root_path = 'videos'
    healthy_videos_root_path = os.path.join(videos_root_path, 'healthy')
    unhealthy_videos_root_path = os.path.join(videos_root_path, 'unhealthy')
    # find all generate folders
    healthy_folders = [os.path.join(healthy_videos_root_path, folder) for folder in os.listdir(healthy_videos_root_path) if os.path.isdir(os.path.join(healthy_videos_root_path, folder))]
    unhealthy_folders = [os.path.join(unhealthy_videos_root_path, folder) for folder in os.listdir(unhealthy_videos_root_path) if os.path.isdir(os.path.join(unhealthy_videos_root_path, folder))]

    # generate healthy classifier data
    output_folder = 'classifier_data'
    # remove the old data
    if os.path.exists(os.path.join(output_folder, 'classifier_data_{}.csv'.format(0))):
        os.remove(os.path.join(output_folder, 'classifier_data_{}.csv'.format(0)))
        print('Remove old healthy data')
    if os.path.exists(os.path.join(output_folder, 'classifier_data_{}.csv'.format(1))):
        os.remove(os.path.join(output_folder, 'classifier_data_{}.csv'.format(1)))
        print('Remove old unhealthy data')

    healthy_output_folder = os.path.join(videos_root_path, 'classifier_data_healthy')
    for folder in healthy_folders:
        trajectories_path = os.path.join(folder, 'trajectories', 'without_gaps.npy')
        generate_classifer_data(trajectories_path, output_folder, label=0)

    for folder in unhealthy_folders:
        trajectories_path = os.path.join(folder, 'trajectories', 'without_gaps.npy')
        generate_classifer_data(trajectories_path, output_folder, label=1)

    # utilize the classifier data to organize the data for classifier
    classifier_data_healthy = os.path.join(output_folder, 'classifier_data_{}.csv'.format(0))
    classifier_data_unhealthy = os.path.join(output_folder, 'classifier_data_{}.csv'.format(1))
    X_train, Y_train, X_test, Y_test = reorganize_data_for_classifer(classifier_data_healthy, classifier_data_unhealthy)

    # train and eval models
    models = train_v2(X_train, Y_train)
    eval_models(models, X_test, Y_test)

    # Save models
    save_models(models, 'output_models')