if [ $(hostname) = "ED716" ]; then
	export CUDA_VISIBLE_DEVICES=0
elif [ $(hostname) = "esc4000-g4" ]; then
	export CUDA_VISIBLE_DEVICES=1
fi

## Create `data.TD_RvNN.vol_5000.txt` from my datasets
#python preprocess.py --dataset twitter15 --create_from_my_dataset
#python preprocess.py --dataset twitter16 --create_from_my_dataset
#python preprocess.py --dataset semeval2019 --create_from_my_dataset

###################
## Top-Down RvNN ##
###################
folds=$(seq 0 4)
for dataset in Twitter15 Twitter16
do
	for fold in $folds
	do
		python Main_TD_RvNN.py \
			--fold $fold \
			--obj $dataset \
			--Nepoch 100
	done
done