import pandas as pd
import numpy as np
from MachineLearning.preprocessing import LabelEncoder, OrdinalEncoder

DATASETS_PATH = {
	'iris_dataset': 'MachineLearning//all_data//iris_dataset.txt',
	'boston_dataset': 'MachineLearning//all_data//boston_dataset.txt',
	'breast_cancer': 'MachineLearning//all_data//breast_cancer_dataset.txt',
	'wine_dataset': 'MachineLearning//all_data//wine_dataset.txt'
}


def extract_data(path, x_idx = 0, y_idx = -1):
	with open(path, 'r') as f:
		dataset = f.readlines()
		X, y = [], []
		for line in dataset[1:]:
			sample = line.split(',')
			if x_idx == 0:
				X.append(sample[x_idx:-1])
			else:
				X.append(sample[x_idx:-1] + [sample[-1][:-1]])
			y.append(sample[y_idx])
		X = pd.DataFrame(X)
		y = pd.Series(y)
	f.close()
	return X, y, dataset


def label_encode_data(y):
	enc = LabelEncoder(dtype=np.int)
	y_enc = enc.fit_transform(np.array(y).reshape(-1, 1))
	return y_enc, enc.labels_

def format_data(X, y, feature_names, target_name, labels=None):
	data = {
	'data': np.array(X),
	'target': np.array(y).reshape(-1, 1),
	'feature_names': feature_names,
	'target_name': target_name
	}
	if labels is not None:
		data['labels'] = labels
	return data


def load_iris():
	X, y, dataset = extract_data(DATASETS_PATH['iris_dataset'], x_idx=0, y_idx=-1)
	X = X.astype(np.float)
	y = y.str[:-1][:-1]

	y_enc, labels = label_encode_data(y)
	
	columns = np.array(dataset[0].split(','))
	return format_data(X, y_enc, columns[:-1], columns[-1][:-1], labels)


def load_boston():
	X, y, dataset = extract_data(DATASETS_PATH['boston_dataset'], x_idx = 0, y_idx=-1)
	y = y.astype(np.float)
	X.iloc[:, 3] = X.iloc[:, 3].str.split('"').str[1].astype(int)
	X = X.astype(np.float)

	columns = np.array(dataset[0].split(','))
	return format_data(X, y, columns[:-1], columns[-1][:-1])


def load_breast_cancer():
	X, y, dataset = extract_data(DATASETS_PATH['breast_cancer'], x_idx=2, y_idx=1)
	X = X.astype(np.float)

	y_enc, labels = label_encode_data(y)

	columns = dataset[0].split(',')
	columns = columns[:-1] + [columns[-1][:-1]]
	columns = np.array(columns)
	return format_data(X, y_enc, columns[2:], columns[1], labels)


def load_wine():
	X, y, dataset = extract_data(DATASETS_PATH['wine_dataset'], x_idx=1, y_idx=0)
	X = X.astype(np.float)

	y_enc, labels = label_encode_data(y)

	columns = dataset[0].split(',')
	columns = columns[:-1] + [columns[-1][:-1]]
	columns = np.array(columns)
	return format_data(X, y_enc, columns[1:], columns[0], labels)
