# Experiment all tricks with center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
# python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('/root/workplace/dataset/')" OUTPUT_DIR "('/root/workplace/reid-strong-baseline/market1501/prw_gt')"

python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('/root/workplace/dataset/')" OUTPUT_DIR "('/root/workplace/re-Id/reid-strong-baseline/market1501/cuhk_gt')"


# 
# python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('prw')" DATASETS.ROOT_DIR "('/root/workplace/dataset/')" OUTPUT_DIR "('/root/workplace/re-Id/reid-strong-baseline/prw/prwgallery_det2')"

# python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('cuhk')" DATASETS.ROOT_DIR "('/root/workplace/dataset/')" OUTPUT_DIR "('/root/workplace/re-Id/reid-strong-baseline/cuhk/prw2cuhkgallery_det_veryweak')"
# 
# python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('prw')" DATASETS.ROOT_DIR "('/root/workplace/dataset/')" OUTPUT_DIR "('/root/workplace/re-Id/reid-strong-baseline/prw/cuhk2prwgallery_det_weak')"