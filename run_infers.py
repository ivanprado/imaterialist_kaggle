import resource

from infer import infer_runner

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print("LIMIT before: {}".format(rlimit))
resource.setrlimit(resource.RLIMIT_NOFILE, (40000, rlimit[1]))
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print("LIMIT after: {}".format(rlimit))

model_files = [
  #"runs/"+ "May11_09-34-42_cs231n-1resnet101-bs-64-clr1e-5-0.1-mom0.9-imgsize-224-pos-weight3" + "/model_best.pth.tar", # 0.599
  #"runs/"+ "May16_13-38-21_cs231n-1resnet101-bs-64-lr0.01-mom0.9-wd4e-4-pos-weight3" + "/model_best.pth.tar", # 0.603
  #"runs/"+ "May20_08-35-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3" + "/model_best.pth.tar", # 0.6036
  #"runs/"+ "May20_22-27-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4" + "/model_best.pth.tar", # 0.6038
  #"runs/"+ "May21_17-04-17_cs231n-1sexception-bs-38-lr0.1-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar", # 0.6541
  #"runs/"+ "May21_17-04-17_cs231n-1sexception-bs-38-lr0.1-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar", # 0.6541
  #"runs/"+ "May21_22-09-11_cs231n-1sexception-bs-38-clr0.001-0.01-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar", # 0.6539
  #"runs/"+ "May22_07-17-10_cs231n-1xception-bs-32-lr0.5-mom0.5-wd1e-5-cutout4-minscale0.4-rota15" + "/model_best.pth.tar", # 0.6491, PW1!
  #"runs/"+ "May23_21-11-42_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas" + "/model_best.pth.tar", # 0.655, PW1
  #"runs/"+ "May24_07-07-00_cs231n-1se_resnext50_32x4d-bs-64-lr0.0006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas" + "/model_best.pth.tar", # 0.6556, PW1
  #"runs/"+ "May24_16-06-22_cs231n-1se_resnext50_32x4d-bs-64-lr0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-label-smoothing0.1" + "/model_best.pth.tar", # 0.6528, PW1
  "runs/"+ "May28_10-50-56_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes50-trainval" + "/model_best.pth.tar", # 0.660, PW1
]

train_samples_limit = 30000
tta = False
batch_size = 64
n_models = len(model_files)
for i, model_file in enumerate(model_files):
  print("Inference ({}/{}):".format(i+1, n_models))
  print("------------------")
  for set_type in ['train', 'validation', 'test']:
    print("Set: {}".format(set_type))
    img_set_folder = "data/" + set_type
    if set_type == 'train':
      samples_limit = train_samples_limit
    else:
      samples_limit = None
    infer_runner(img_set_folder, model_file, samples_limit=samples_limit, tta=tta, batch_size=batch_size)