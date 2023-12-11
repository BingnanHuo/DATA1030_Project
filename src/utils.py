# Data analysis
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import uniform, loguniform
from skopt.space import Real, Categorical, Integer
import pickle


# Preprocessing & Splitting
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold

# Modeling
from skopt import BayesSearchCV 
import xgboost
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score, classification_report, confusion_matrix



def sample_dataset(X, y, group, sample_size=0.1, random_state=42):
    np.random.seed(random_state)
    sampled_groups = np.random.choice(group.unique(), int(len(group.unique())*sample_size), replace=False)
    mask = group.isin(sampled_groups)
    return X[mask], y[mask], group[mask]


def vote_pred(y_true, y_pred, group):
    combined_data = pd.DataFrame({
        'group': group.values,
        'true_labels': y_true,
        'predictions': y_pred
    })
    def most_common_label(series):
        return Counter(series).most_common(1)[0][0]
    aggregated_data = combined_data.groupby('group').agg(most_common_label)
    y_true_aggregated = aggregated_data['true_labels']
    y_pred_aggregated = aggregated_data['predictions']
    return y_true_aggregated, y_pred_aggregated


def downsample_classes(y, group, random_state=42):

    data = pd.DataFrame({'label': y, 'group': group}).reset_index(drop=True)
    min_label_count = Counter(y).most_common()[-1][1]

    # Collect indices of each label
    label_indices = {label: data[data['label'] == label].index for label in set(y)}
    downsampled_indices = []

    for label, indices in label_indices.items():
        # Filter groups for this label
        groups_in_label = data.loc[indices, 'group'].unique()

        # Shuffle the groups to randomize selection
        np.random.seed(random_state)
        np.random.shuffle(groups_in_label)

        selected_indices = []

        # Select groups until we reach or exceed the minimum count
        for group in groups_in_label:
            group_indices = data[(data['group'] == group) & (data['label'] == label)].index
            if len(selected_indices) + len(group_indices) <= min_label_count:
                selected_indices.extend(group_indices)
            else:
                break  # Stop if adding this group exceeds the minimum count

        downsampled_indices.extend(selected_indices)

    return downsampled_indices


def oversampled_classes(y, y_samp):
    #print('Ensuring minority classes are sampled enough.')
    small_class_labels = [8,9,10]
    check_props = np.zeros(len(small_class_labels))
    for i in range(len(small_class_labels)):
        this_label = small_class_labels[i]
        # if proportion of small class in sampled data <= that in original data, false
        sample_prop = np.round((y_samp == this_label).mean(),3)
        orig_prop = np.round((y == this_label).mean(),3)
        if sample_prop >= orig_prop.mean():
            check_props[i] = True
        else:
            check_props[i] = False
    return check_props.mean() == 1
    
        
def print_metrics(y_true, y_pred):
    print('Accuracy: ', np.round(accuracy_score(y_true, y_pred), 3))
    print('Precision: ', np.round(precision_score(y_true, y_pred, average='weighted'), 3))
    print('Recall: ', np.round(recall_score(y_true, y_pred, average='weighted'), 3))
    print('F1: ', np.round(f1_score(y_true, y_pred, average='weighted'), 3))


def plot_confusion_matrices(y_trues, y_preds, titles):
    num_plots = len(y_trues)
    fig, axes = plt.subplots(1, num_plots, figsize=(8*num_plots, 6), tight_layout=True)
    # In case of a single plot, axes is not an array, so we wrap it in a list
    if num_plots == 1:
        axes = [axes]
    for i in range(num_plots):
        cm = confusion_matrix(y_trues[i], y_preds[i])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    plt.show()

    