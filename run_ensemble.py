from ensemble import PositiveMeanEnsemble, MeanEnsemble, BestPerClassEnsemble

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
  "runs/"+ "May28_10-50-56_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes50-trainval" + "/model_best.pth.tar", # 0.660, PW1
  "runs/" + "May28_16-26-56_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6626, PW1
  "runs/" + "May28_19-19-15_cs231n-1se_resnext50_32x4d-bs-64-clr0.0006-0.00006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6633, PW1
  "runs/" + "May28_21-27-57_cs231n-1sexception-bs-38-clr0.1-0.01-0.001-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6621, PW1
  "runs/" + "May29_06-04-06_cs231n-1se_resnext50_32x4d-bs-64-clr0.006-0.0006-mom0.9-wd1e-5-scale0.3-0.6-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6631, PW1

]

ensemble_selection = [
  "runs/"+ "May16_13-38-21_cs231n-1resnet101-bs-64-lr0.01-mom0.9-wd4e-4-pos-weight3" + "/model_best.pth.tar", # 0.603
  "runs/"+ "May20_22-27-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4" + "/model_best.pth.tar", # 0.6038
  "runs/"+ "May21_17-04-17_cs231n-1sexception-bs-38-lr0.1-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar", # 0.6541
  "runs/" + "May24_07-07-00_cs231n-1se_resnext50_32x4d-bs-64-lr0.0006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas" + "/model_best.pth.tar", # 0.6556, PW1
  "runs/"+ "May24_16-06-22_cs231n-1se_resnext50_32x4d-bs-64-lr0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-label-smoothing0.1" + "/model_best.pth.tar", # 0.6528, PW1
  "runs/"+ "May28_10-50-56_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes50-trainval" + "/model_best.pth.tar", # 0.660, PW1
  "runs/" + "May28_19-19-15_cs231n-1se_resnext50_32x4d-bs-64-clr0.0006-0.00006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6633, PW1
  "runs/" + "May28_21-27-57_cs231n-1sexception-bs-38-clr0.1-0.01-0.001-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6621, PW1
  "runs/" + "May29_06-04-06_cs231n-1se_resnext50_32x4d-bs-64-clr0.006-0.0006-mom0.9-wd1e-5-scale0.3-0.6-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6631, PW1
]

ensemble_diverse = [
  "runs/"+ "May20_22-27-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4" + "/model_best.pth.tar", # 0.6038
  "runs/"+ "May28_10-50-56_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes50-trainval" + "/model_best.pth.tar", # 0.660, PW1
  "runs/" + "May28_19-19-15_cs231n-1se_resnext50_32x4d-bs-64-clr0.0006-0.00006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar", # 0.6633, PW1
]


set_type = 'test'
thresholds_type = 'validation'
tta = True
ensemble = MeanEnsemble(ensemble_selection, set_type, thresholds_type, tta)
#ensemble = BestPerClassEnsemble(ensemble_selection, set_type, thresholds_type, tta, top=3)
ensemble.infer()

#learners_data = LearnersData(ensemble_all, set_type, thresholds_type, tta)
#metalearner = MetaLearner(learners_data, batch_size=1024)
#metalearner.train(100000)
