# !bin/bash

# IEMOCAP
python train.py --hidden_dim=300 --learning_rate=0.00017 --alpha=0.037 --beta=0.012 --tensor_rank=11 --dataset=iemocap --start_mod=4 --mmcosine --modulation --normalize --modalities=atv --batch_size=10 --epoch=50 --early_stopping=20 --seed=12 --mmt_nlayers=2 
# MELD
python train.py --hidden_dim=200 --learning_rate=0.00013 --alpha=0.42 --beta=0.55 --tensor_rank=6 --dataset=meld --start_mod=4 --mmcosine --modulation --normalize --modalities=atv --batch_size=10 --epoch=50 --early_stopping=20 --seed=12 --mmt_nlayers=2
