from __future__ import print_function, division

import matplotlib.pyplot as plt

from infer import infer_runner
import resource
plt.ion()

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print("LIMIT before: {}".format(rlimit))
resource.setrlimit(resource.RLIMIT_NOFILE, (40000, rlimit[1]))
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print("LIMIT after: {}".format(rlimit))

model_file= "runs/"+ "May11_05-47-56_cs231n-1resnet101-bs-64-clr5e-6-0.05-mom0.9-imgsize-224" + "/model_best.pth.tar" # 0.502
model_file= "runs/"+ "May11_09-34-42_cs231n-1resnet101-bs-64-clr1e-5-0.1-mom0.9-imgsize-224-pos-weight3" + "/model_best.pth.tar" # 0.599
model_file = "runs/"+ "May16_13-38-21_cs231n-1resnet101-bs-64-lr0.01-mom0.9-wd4e-4-pos-weight3" + "/model_best.pth.tar" # 0.603
model_file = "runs/"+ "May17_17-12-16_cs231n-1xception-bs-64-lr0.045-mom0.9-wd1e-5-pos-weight3-since-block4" + "/model_best.pth.tar" # 0.568 INVALIDO
model_file = "runs/"+ "May17_16-04-29_cs231n-1xception-bs-64-lr0.045-mom0.9-wd1e-5-pos-weight3-just-fc" + "/model_best.pth.tar" # 0.4507739507786539 Sólo la fc entrenada un poquejo. INVALIDO
model_file = "runs/"+ "May18_07-37-59_cs231n-1xception-bs-64-lr0.045-mom0.9-wd1e-5-pos-weight3-just-fc" + "/model_best.pth.tar" # 0.45
model_file = "runs/"+ "May20_08-35-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3" + "/model_best.pth.tar" # 0.6036
model_file = "runs/"+ "May20_22-27-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4" + "/model_best.pth.tar" # 0.6038
model_file = "runs/"+ "May21_17-04-17_cs231n-1sexception-bs-38-lr0.1-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar" # 0.6541
model_file = "runs/"+ "May21_17-04-17_cs231n-1sexception-bs-38-lr0.1-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar" # 0.6541
model_file = "runs/"+ "May21_22-09-11_cs231n-1sexception-bs-38-clr0.001-0.01-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar" # 0.6539
model_file = "runs/"+ "May23_21-11-42_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas" + "/model_best.pth.tar" # 0.655, PW1
model_file = "runs/"+ "May24_07-07-00_cs231n-1se_resnext50_32x4d-bs-64-lr0.0006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas" + "/model_best.pth.tar" # 0.6556, PW1


model_type="se_resnext50_32x4d"
img_set_folder = "data/validation"
infer_runner(img_set_folder, model_file, samples_limit=500, tta=True, batch_size=4, write=False)
