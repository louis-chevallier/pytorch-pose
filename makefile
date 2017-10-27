
export ROOT_DIR=${PWD}
pref = $(dir $2)$1-$(notdir $2)
DATE=`date +%Y-%m-%d_%Hh%Mm%Ss`
define cr

endef

GPUS=0,1,2,3,4,5,6,7
GPUS=4

$(warning cuda : $(CUDA_VISIBLE_DEVICES))

PP=$(CURDIR)/../../../util_lc/Utils
COMMA=,
NGPU=$(words $(subst $(COMMA), ,$(CUDA_VISIBLE_DEVICES)))
$(warning $(NGPU))

BATCH=$(shell expr 30 '*'  $(NGPU))
$(warning $(BATCH))

validate :
	-CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$(PP) python example/mpii.py -a hg --stacks 8 --blocks 1 --checkpoint checkpoint/mpii/hg8 --test-batch 1 --resume checkpoint/mpii/hg8/model_best.pth.tar -e 2>&1 | tee  $(DATE)-$@-hg8.trc

www :
	echo 

start1 :
	-CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=.:$(PP) python example/mpii.py -a hg --stacks 8 --blocks 1 --checkpoint checkpoint/mpii/hg8_1 -j 8 --train-batch 80


predict :
	-CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$(PP) python example/mpii.py -a hg --stacks 8 --blocks 1 --checkpoint checkpoint/mpii/hg8 --resume checkpoint/mpii/hg8/model_best.pth.tar --test sakura.jpg 2>&1 | tee  $(DATE)-$@-hg8.trc


start :
#	-CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=.:$(PP) python example/mpii.py -a hg --stacks 8 --blocks 1 --checkpoint checkpoint/mpii/hg8 -j 24 --train-batch 80  2>&1   | tee  $(DATE)-train-mpii_hg8.trc
#	-CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=.:$(PP) python example/mpii.py -a hg --stacks 8 --blocks 1 --checkpoint checkpoint/mpii/hg8_a -j 24 --train-batch 80 --sigma 4 --sigma-decay 0.98 2>&1   | tee  $(DATE)-train-mpii_hg8_a.trc
#	-CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=.:$(PP) python example/mpii.py -a hg --stacks 8 --blocks 2 --checkpoint checkpoint/mpii/hg8_b2 -j 24 --train-batch 40 2>&1   | tee  $(DATE)-train-mpii_hg8_b2.trc
	date
	-PYTHONPATH=.:$(PP):./pose python example/mpii.py -a hg --stacks 2 --blocks 1 --checkpoint checkpoint/mpii/hg4_b2 -j 12 --train-batch $(BATCH) 2>&1 --epochs 1  | tee  $(DATE)-train-mpii_hg4_b2.trc
	date

startmany :
	date
	-CUDA_VISIBLE_DEVICES=0 make start
	date
	-CUDA_VISIBLE_DEVICES="0,1" make start
	date
	-CUDA_VISIBLE_DEVICES="0,1,2,3,4" make start



eval :
	CUDA_VISIBLE_DEVICES=0,1  PYTHONPATH=.:$(PP) python example/mpii.py -a hg --stacks 8 --blocks 1 --checkpoint checkpoint/mpii/hg8 --resume checkpoint/mpii/hg8/model_best.pth.tar  -e  | tee  $(DATE)-eval-mpii_hg8_b4.trc 

sub :
	echo CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=.:$(PP) python example/mpii.py -a hg --stacks 8 --blocks 1 --checkpoint checkpoint/mpii/hg8 --resume checkpoint/mpii/hg8/model_best.pth.tar  -e | qsub -q gpu.q -l gpu=4 -cwd

#python example/mpii.py -a hg --stacks 8 --blocks 1 --checkpoint checkpoint/mpii/hg8 --resume checkpoint/mpii/hg8/model_best.pth.tar y -ngpu 4
