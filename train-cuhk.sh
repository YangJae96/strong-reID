# Experiment all tricks without center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# without center loss




##### Label smooth CE + triplet #####
# python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('cuhk')" DATASETS.ROOT_DIR "('/root/workplace/dataset/')" OUTPUT_DIR "('/root/workplace/re-Id/reid-strong-baseline/cuhk/cuhk2gt_no_center2')"
#####################################################################################

##### Only crossEntropy #####
# python3 tools/train.py --config_file='configs/softmax.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('cuhk')" DATASETS.ROOT_DIR "('/root/workplace/dataset/')" OUTPUT_DIR "('/root/workplace/re-Id/reid-strong-baseline/cuhk/cuhk2gt_onlycross-Entropy')"
# #####################################################################################


#### CrossEntropy + Supervised Contrastive loss #####
python3 tools/train.py --config_file='configs/softmax_supcon.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('cuhk')" DATASETS.ROOT_DIR "('/root/workplace/dataset/')" OUTPUT_DIR "('/root/workplace/re-Id/reid-strong-baseline/cuhk/cuhk2gt/supconLoss-crossEntropy_conv1freeze')"
#####################################################################################

