from BlogTutorials.pyimagesearch.utils.ranked import rank5_accuracy
import argparse
import pickle
import h5py
database_path = r'D:\matth\Documents\projects\python\vgg16_feature_extraction\flowers-17\flowers_features.hdf5'
model_path = r'D:\matth\Documents\projects\python\models\vgg16_imagenetTransfer_flowers17_log_regression_class\model.sav'

print('[info] loading model')
model = pickle.loads(open(model_path, "rb").read())

# load extracted feature dataset
db = h5py.File(database_path, "r")
i = int(db["labels"].shape[0] * 0.75 )

print("[info] predicting...")
preds = model.predict_proba(db["features"][i:])
(rank1, rank5) = rank5_accuracy(preds, db["labels"][i:])
print("rank1: {:.2f}%".format(rank1 * 100) )
print("rank5: {:.2f}%".format(rank5 * 100) )

db.close()