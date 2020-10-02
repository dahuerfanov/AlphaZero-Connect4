import pickle


def save_obj(obj, work_path, name):
    with open(work_path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(work_path, name):
    with open(work_path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
