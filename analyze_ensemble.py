import numpy as np

from ensemble import PositiveMeanEnsemble, MeanEnsemble, LearnersData
import matplotlib.pyplot as plt

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
  "runs/"+ "May24_16-06-22_cs231n-1se_resnext50_32x4d-bs-64-lr0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-label-smoothing0.1" + "/model_best.pth.tar", # 0.6528, PW1
]

ensemble_4_mix = [
  "runs/"+ "May16_13-38-21_cs231n-1resnet101-bs-64-lr0.01-mom0.9-wd4e-4-pos-weight3" + "/model_best.pth.tar", # 0.603
  "runs/"+ "May20_22-27-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4" + "/model_best.pth.tar", # 0.6038
  "runs/"+ "May21_17-04-17_cs231n-1sexception-bs-38-lr0.1-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar", # 0.6541
  "runs/" + "May24_07-07-00_cs231n-1se_resnext50_32x4d-bs-64-lr0.0006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas" + "/model_best.pth.tar", # 0.6556, PW1
  #"runs/"+ "May24_16-06-22_cs231n-1se_resnext50_32x4d-bs-64-lr0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-label-smoothing0.1" + "/model_best.pth.tar", # 0.6528, PW1
]


set_type = 'validation'
thresholds_type = 'validation'
tta = False
ensemble = LearnersData(ensemble_4_mix, set_type, thresholds_type, tta)
#  N, N X C, L x N x C, L x C
image_ids, labels, confidences_LNC, thresholds = ensemble.set_data


confidences_NLC = confidences_LNC.transpose((1, 0, 2)) # N x L x C
pred_NLC = confidences_NLC > thresholds # N x L x C

pred_LNC = pred_NLC.transpose((1, 0 , 2))
tp = pred_LNC * labels
fp = pred_LNC * (1 - labels)
tn = (1 - pred_LNC) * (1 - labels)
fn = (1 - pred_LNC) * (labels)

tp_agg = tp.sum(axis=(1,2))
fp_agg = fp.sum(axis=(1,2))
tn_agg = tn.sum(axis=(1,2))
fn_agg = fn.sum(axis=(1,2))

precision = tp_agg / (tp_agg + fp_agg)
recall = tp_agg / (tp_agg + fn_agg)
f1_score = (2 * precision * recall) / (precision + recall)

print(f1_score)

fp[:,:,14] = 0
fn[:,:,14] = 0
tp[:,:,14] = 0
tn[:,:,14] = 0

tp_agg = tp.sum(axis=(1,2))
fp_agg = fp.sum(axis=(1,2))
tn_agg = tn.sum(axis=(1,2))
fn_agg = fn.sum(axis=(1,2))

precision = tp_agg / (tp_agg + fp_agg)
recall = tp_agg / (tp_agg + fn_agg)
f1_score = (2 * precision * recall) / (precision + recall)

print(f1_score)


fails = fp + fn # L x N x C
freqs = np.load("freqs.npy")

idx_sort_by_freq = np.argsort(freqs)
sorted_fails = fails[:, :, idx_sort_by_freq]
per_class = sorted_fails.sum(axis=1) # L x C
per_class[per_class<1000] = 0
per_class_unsorted = fails.sum(axis=1)

strange = np.where(per_class>1000)[1][0]
class_strange = idx_sort_by_freq[strange]
value_strange = per_class_unsorted[0][class_strange]
print(idx_sort_by_freq[strange], value_strange)

#print(np.where(per_class_unsorted>1000))

plt.figure(figsize=(10,10))
plt.plot(freqs[idx_sort_by_freq] * 2500)
for i in range(per_class.shape[0]):
  plt.plot(per_class[i,:], marker="x", linewidth=0)

plt.savefig("fails_per_model.png")



#learners_data = LearnersData(ensemble_all, set_type, thresholds_type, tta)
#metalearner = MetaLearner(learners_data, batch_size=1024)
#metalearner.train(100000)
