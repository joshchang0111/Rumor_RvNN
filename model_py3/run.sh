folds=$(seq 0 4)

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