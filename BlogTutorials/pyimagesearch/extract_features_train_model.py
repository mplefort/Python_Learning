from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

database_path = r'D:\matth\Documents\projects\python\vgg16_feature_extraction\flowers-17\flowers_features.hdf5'
output_model_path = r'D:\matth\Documents\projects\python\models\vgg16_imagenetTransfer_flowers17_log_regression_class\model.sav'
jobnum = 2

# open HDF5 database for reading then determin the index of the training and testing split, provided that
# this data was already shuffled *PRIOR* to writing to disk
db = h5py.File(database_path, "r")
i = int(db["labels"].shape[0]*0.75)

# define the set of parameters that we want to tun then start a grid search where we evaluate our model for each value of C
print("[info] tuning hyperparmaeter...")
params = {"C": [0.01]}
model = GridSearchCV(LogisticRegression(), param_grid=params, cv=3, n_jobs=jobnum)
model.fit(db["features"][:i], db["labels"][:i])
print("[info] best hyperparams: {}".format(model.best_params_))

# evaluate model
print("[info] evaluating model...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:],
                            preds,
                            target_names=db["label_names"]))

# save model
print("[info] saving model...")
f = open(output_model_path, "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()
db.close()
