from ensemble import MeanEnsemble, QuorumEnsemble, PositiveMeanEnsemble, MetaLearner, LearnersData

ensemble_all = [
  "runs/"+ "May11_09-34-42_cs231n-1resnet101-bs-64-clr1e-5-0.1-mom0.9-imgsize-224-pos-weight3" + "/model_best.pth.tar", # 0.599
  "runs/"+ "May16_13-38-21_cs231n-1resnet101-bs-64-lr0.01-mom0.9-wd4e-4-pos-weight3" + "/model_best.pth.tar", # 0.603
  "runs/"+ "May20_08-35-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3" + "/model_best.pth.tar", # 0.6036
  "runs/"+ "May20_22-27-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4" + "/model_best.pth.tar", # 0.6038
  "runs/"+ "May21_17-04-17_cs231n-1sexception-bs-38-lr0.1-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar", # 0.6541
  "runs/"+ "May21_22-09-11_cs231n-1sexception-bs-38-clr0.001-0.01-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar", # 0.6539
  #"runs/"+ "May22_07-17-10_cs231n-1xception-bs-32-lr0.5-mom0.5-wd1e-5-cutout4-minscale0.4-rota15" + "/model_best.pth.tar" # 0.6491, PW1!
]

ensemble_3_mix = [
  "runs/"+ "May16_13-38-21_cs231n-1resnet101-bs-64-lr0.01-mom0.9-wd4e-4-pos-weight3" + "/model_best.pth.tar", # 0.603
  "runs/"+ "May20_22-27-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4" + "/model_best.pth.tar", # 0.6038
  "runs/"+ "May21_17-04-17_cs231n-1sexception-bs-38-lr0.1-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar", # 0.6541
]



set_type = 'validation'
thresholds_type = 'validation'
tta = False
#ensemble = PositiveMeanEnsemble(ensemble_all, set_type, thresholds_type, tta, positive_threshold=0.0)
#ensemble.infer()

learners_data = LearnersData(ensemble_all, set_type, thresholds_type, tta)
metalearner = MetaLearner(learners_data, batch_size=1024)
metalearner.train(100000)
