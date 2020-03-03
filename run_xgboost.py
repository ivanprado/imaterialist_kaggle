import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from ensemble import PositiveMeanEnsemble, MeanEnsemble, BestPerClassEnsemble, LearnersData
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost.sklearn import XGBClassifier, XGBRegressor

from measures import just_f1

ensemble_all = [
  "runs/"+ "May11_09-34-42_cs231n-1resnet101-bs-64-clr1e-5-0.1-mom0.9-imgsize-224-pos-weight3" + "/model_best.pth.tar", # 0.599
  "runs/"+ "May16_13-38-21_cs231n-1resnet101-bs-64-lr0.01-mom0.9-wd4e-4-pos-weight3" + "/model_best.pth.tar", # 0.603
  "runs/"+ "May20_08-35-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3" + "/model_best.pth.tar", # 0.6036
  "runs/"+ "May20_22-27-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4" + "/model_best.pth.tar", # 0.6038
  "runs/"+ "May21_17-04-17_cs231n-1sexception-bs-38-lr0.1-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar", # 0.6541
  "runs/"+ "May21_22-09-11_cs231n-1sexception-bs-38-clr0.001-0.01-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar", # 0.6539
  "runs/"+ "May22_07-17-10_cs231n-1xception-bs-32-lr0.5-mom0.5-wd1e-5-cutout4-minscale0.4-rota15" + "/model_best.pth.tar", # 0.6491, PW1!
  "runs/" + "May23_21-11-42_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas" + "/model_best.pth.tar", # 0.655, PW1
  "runs/" + "May24_07-07-00_cs231n-1se_resnext50_32x4d-bs-64-lr0.0006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas" + "/model_best.pth.tar", # 0.6556, PW1
  #"runs/"+ "May24_16-06-22_cs231n-1se_resnext50_32x4d-bs-64-lr0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-label-smoothing0.1" + "/model_best.pth.tar", # 0.6528, PW1
  #"runs/"+ "May28_10-50-56_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes50-trainval" + "/model_best.pth.tar", # 0.660, PW1
  #"runs/" + "May28_16-26-56_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6626, PW1
  #"runs/" + "May28_19-19-15_cs231n-1se_resnext50_32x4d-bs-64-clr0.0006-0.00006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6633, PW1
  #"runs/" + "May28_21-27-57_cs231n-1sexception-bs-38-clr0.1-0.01-0.001-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6621, PW1
  #"runs/" + "May29_06-04-06_cs231n-1se_resnext50_32x4d-bs-64-clr0.006-0.0006-mom0.9-wd1e-5-scale0.3-0.6-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6631, PW1

  #"runs/" + "May29_19-25-32_cs231n-1se_resnext101_32x4d-bs-64-lr0.006-mom0.9-wd1e-5-minscale0.3-rota15-cas-best-classes15-trainval" + "/model_best.pth.tar",  # 0.6641, PW1
  #"runs/" + "May29_15-14-04_cs231n-1se_resnext101_32x4d-bs-64-lr0.06-mom0.9-wd1e-5-minscale0.3-rota15-cas-best-classes15-trainval" + "/model_best.pth.tar",  # 0.6637, PW1

  #"runs/" + "May28_16-26-56_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar",  # 0.6626, PW1
  #"runs/" + "May30_16-09-41_cs231n-1se_resnext50_32x4d-bs-64-lr0.006-mom0.9-wd1e-5-minscale0.3-rota10-ratio0.9-1.1-cas-best-classes15-trainval" + "/model_best.pth.tar",# 0.6636
]

ensemble_selection = [
  #"runs/"+ "May16_13-38-21_cs231n-1resnet101-bs-64-lr0.01-mom0.9-wd4e-4-pos-weight3" + "/model_best.pth.tar", # 0.603
  #"runs/"+ "May20_22-27-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4" + "/model_best.pth.tar", # 0.6038
  #"runs/"+ "May21_17-04-17_cs231n-1sexception-bs-38-lr0.1-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar", # 0.6541
  #"runs/" + "May24_07-07-00_cs231n-1se_resnext50_32x4d-bs-64-lr0.0006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas" + "/model_best.pth.tar", # 0.6556, PW1
  #"runs/"+ "May24_16-06-22_cs231n-1se_resnext50_32x4d-bs-64-lr0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-label-smoothing0.1" + "/model_best.pth.tar", # 0.6528, PW1
  #"runs/"+ "May28_10-50-56_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes50-trainval" + "/model_best.pth.tar", # 0.660, PW1
  #"runs/" + "May28_19-19-15_cs231n-1se_resnext50_32x4d-bs-64-clr0.0006-0.00006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6633, PW1
  "runs/" + "May28_21-27-57_cs231n-1sexception-bs-38-clr0.1-0.01-0.001-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6621, PW1
  "runs/" + "May29_06-04-06_cs231n-1se_resnext50_32x4d-bs-64-clr0.006-0.0006-mom0.9-wd1e-5-scale0.3-0.6-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6631, PW1

  "runs/" + "May29_19-25-32_cs231n-1se_resnext101_32x4d-bs-64-lr0.006-mom0.9-wd1e-5-minscale0.3-rota15-cas-best-classes15-trainval" + "/model_best.pth.tar",  # 0.6641, PW1
  "runs/" + "May29_15-14-04_cs231n-1se_resnext101_32x4d-bs-64-lr0.06-mom0.9-wd1e-5-minscale0.3-rota15-cas-best-classes15-trainval" + "/model_best.pth.tar",  # 0.6637, PW1

  #"runs/" + "May28_16-26-56_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar",  # 0.6626, PW1
  "runs/" + "May30_16-09-41_cs231n-1se_resnext50_32x4d-bs-64-lr0.006-mom0.9-wd1e-5-minscale0.3-rota10-ratio0.9-1.1-cas-best-classes15-trainval" + "/model_best.pth.tar",# 0.6636
  #"runs/"+ "May30_18-09-07_cs231n-1se_resnext50_32x4d-bs-64-lr0.0006-mom0.9-wd1e-5-minscale0.3-rota10-ratio0.9-1.1-cas-best-classes10-trainval" + "/model_best.pth.tar" # 0.6628

  #"runs/"+ "May30_20-14-59_cs231n-1se_resnext101_32x4d-bs-38-lr0.0006-mom0.9-wd1e-5-minscale0.3-rota10-ratio0.9-1.1-cas-best-classes-15-trainval" + "/model_best.pth.tar", # 0.6642
]

ensemble_diverse = [
  "runs/"+ "May20_22-27-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4" + "/model_best.pth.tar", # 0.6038
  "runs/"+ "May28_10-50-56_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes50-trainval" + "/model_best.pth.tar", # 0.660, PW1
  "runs/"+ "May28_19-19-15_cs231n-1se_resnext50_32x4d-bs-64-clr0.0006-0.00006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6633, PW1
]


set_type = 'validation'
thresholds_type = 'validation'
tta = False
#ensemble = MeanEnsemble(ensemble_all, set_type, thresholds_type, tta)
#ensemble = BestPerClassEnsemble(ensemble_all, set_type, thresholds_type, tta, top=4)
#ensemble.infer()

#learners_data = LearnersData(ensemble_all, set_type, thresholds_type, tta)
#metalearner = MetaLearner(learners_data, batch_size=1024)
#metalearner.train(100000)

class_freq_on_test = np.load("freq-test.npy")
idx_sorted_freqs = np.flip(np.argsort(class_freq_on_test), axis=0)

data = LearnersData(ensemble_all, set_type, thresholds_type, tta)
data.set_data #the tuple of numpy arrays (image_ids, labels, confidences, thresholds) of size N, N X C, L x N x C, L x C
_, y, X_orig, _ = data.set_data

X_orig = X_orig[:, :, idx_sorted_freqs] #
y = y[:, idx_sorted_freqs]
#X_orig = X_orig[:,:,14:16]
#y = y[:,14:16]

L, N, C = X_orig.shape
X = X_orig.transpose(1, 2, 0).reshape(y.shape[0], -1) # N x C*L



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#base_regr = XGBRegressor(silent=0, n_estimators=300, max_depth=1, )
# base_regr = RandomForestRegressor(max_depth=10, n_estimators=20)
# regr = MultiOutputRegressor(base_regr)
#
# regr.fit(X_train, y_train)
#
# pred = regr.predict(X_test)
# print(pred)
#
# y_true = y_test > 0.3
# y_pred = pred > 0.3
# pred_first = y_pred[:,0]
# print(pred_first.sum())
# print(accuracy_score(y_true, y_pred))
# print(f1_score(y_true, y_pred, average='micro'))
# print(just_f1(y_true.astype(np.float32), pred.astype(np.float32), np.array(0.3)))
#
# regr = regr.estimators_[0]
# y_test = y_test[:,0]
#
# pred = regr.predict(X_test)
# print(pred)
#
# y_true = y_test > 0.3
# y_pred = pred > 0.3
# print(accuracy_score(y_true, y_pred))
# print(f1_score(y_true, y_pred, average='micro'))
# print((pred_first * y_pred).sum())
# print(just_f1(y_true.astype(np.float32), pred.astype(np.float32), np.array(0.3)))



def eval_models(models, X, y, threshold=0.3):
  preds = np.array([m.predict(X) for m in models]).T # N x C
  print(just_f1(y.astype(np.float32), preds.astype(np.float32), np.array(threshold)))

models = []
for i in range(C):
  print("Class {} ...".format(i))
  y_current = y_train[:,i]
  proportion = (y_current < 0.5).sum() / (y_current > 0.5).sum()
  #print("proportion: {}".format(proportion))
  #reg = XGBRegressor(silent=1, n_estimators=120, max_depth=6, n_jobs=7, scale_pos_weight=proportion, learning_rate=0.1, subsample=0.5)
  reg = XGBClassifier(silent=1, n_estimators=60, max_depth=3, n_jobs=7, scale_pos_weight=proportion, learning_rate=0.1,
                     subsample=0.5)
  #reg = RandomForestRegressor(max_depth=5, n_estimators=5)
  reg.fit(X_train, y_current)
  models.append(reg)
  if i%1 == 0 :
    eval_models(models, X_test, y_test[:, :i+1])


eval_models(models, X_test, y_test)


