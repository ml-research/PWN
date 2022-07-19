import pickle


def save_experiment(filepath, experiment):
    with open(filepath + '.pkl', 'wb') as f:
        pickle.dump(experiment, f)


def load_experiment(filepath):
    with open(filepath + '.pkl', 'rb') as f:
        return pickle.load(f)
