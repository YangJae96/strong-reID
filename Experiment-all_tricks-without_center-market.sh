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
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('cuhk')" DATASETS.ROOT_DIR "('/root/workplace/dataset/')" OUTPUT_DIR "('/root/workplace/re-Id/reid-strong-baseline/cuhk/prw2cuhkgallery_det_no_center)"



# python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('cuhk')" DATASETS.ROOT_DIR "('/root/workplace/dataset/')" OUTPUT_DIR "('/root/workplace/re-Id/reid-strong-baseline/cuhk/prw2cuhkgallery_det_weak')"
