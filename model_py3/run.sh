folds=$(seq 0 4)

## Create `data.TD_RvNN.vol_5000.txt` from my datasets
#python preprocess.py --dataset twitter15 --create_from_my_dataset
#python preprocess.py --dataset twitter16 --create_from_my_dataset
#python preprocess.py --dataset semeval2019 --create_from_my_dataset

## Top-Down RvNN ##
for dataset in Twitter15 Twitter16
do
	for fold in $folds
	do
		python Main_TD_RvNN.py \
		--fold $fold \
		--obj $dataset
	done
done