
export ROOT_DIR=${PWD}
pref = $(dir $2)$1-$(notdir $2)
DATE=`date +%Y-%m-%d_%Hh%Mm%Ss`
define cr

endef

PP=$(CURDIR)/../../../util_lc/Utils

#0,1,2,3,4,5,6,7
GPUS=0,2

start1 :
	-CUDA_VISIBLE_DEVICES=$(GPUS) PYTHONPATH=.:$(PP) python example/mpii.py -a hg --stacks 16 --blocks 1 --checkpoint checkpoint/mpii/hg8_1 -j 8 --train-batch 20

