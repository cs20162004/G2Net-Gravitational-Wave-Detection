import pandas as np
from sklearn import model_selection

train_data = np.read_csv("/home/hero/Downloads/Ali/practice/g2net/dt/training_labels.csv")
# test_data = np.read_csv("/home/hero/Downloads/Ali/practice/g2net/dt/sample_submission.csv")

def get_train_file_path(image_id):
    return "/home/hero/Downloads/Ali/practice/g2net/dt/train/{}/{}/{}/{}.npy".format(image_id[0], image_id[1], image_id[2], image_id)

def get_test_file_path(image_id):
    return "/home/hero/Downloads/Ali/practice/g2net/dt/test/{}/{}/{}/{}.npy".format(image_id[0], image_id[1], image_id[2], image_id)

train_data["file_path"] = train_data['id'].apply(get_train_file_path)
# test_data["file_path"] = test_data['id'].apply(get_test_file_path)

Fold = model_selection.StratifiedKFold(n_splits=5, shuffle = True, random_state = 42)
for n, (train_index, val_index) in enumerate(Fold.split(train_data, train_data['target'])):
        train_data.loc[val_index, 'fold'] = int(n)
train_data['fold'] = train_data['fold'].astype(int)

train_data.to_csv("/home/hero/Downloads/Ali/practice/g2net/dt/split_train.csv")
