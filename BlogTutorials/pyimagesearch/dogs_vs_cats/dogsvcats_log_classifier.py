from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import pickle
import h5py

hdf5_dataset = r'D:\matth\Documents\projects\python\datasets\kaggle_dogs_vs_cats\models\resnet50 _feature_extraction\catsvdogresnet.hdf5'
model_path = r'D:\matth\Documents\projects\python\datasets\kaggle_dogs_vs_cats\models\resnet50 _feature_extraction\log_model.pickle'
hyperparamsearch = -1

# load dataset, split for training
db = h5py.File(hdf5_dataset, "r")
i = int(db["labels"].shape[0] * 0.75)

# hyperparam serach
print("[info] seraching hyperparms..")
params = {"C": [0.0001]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=hyperparamsearch)
model.fit(db["features"][:i], y=db["labels"][:i])
print("[INFO] best params: {}".format(model.best_params_))

# generate classification report
print("[info] evaluation...")
preds  = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds, target_names=db["label_names"]))

#  accuracy
acc = accuracy_score(db["labels"][i:], preds)
print("[info] accuracy score: {}".format(acc))

# save model
f = open(model_path, "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

db.close()