set -x
tools/dist_train_local.sh $1/$2 $3 --work-dir $1 2>&1 | tee $1/train.log
# tools/slurm_train.sh pname mytrain $1/$2 $1   2>&1 | tee $1/train.log
