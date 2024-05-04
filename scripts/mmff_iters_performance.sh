for iters in 5 10 20 50 100 150 200 500 1000
do  
    python test.py +experiment=230709-fromtorsion-drugs-7steps-final gpu=0 train.test_mmff=true train.test_mmff_iters=${iters} train.test_starting_point=true
    python test.py +experiment=230709-fromtorsion-drugs-7steps-final gpu=0 train.test_mmff=true train.test_mmff_iters=${iters}
done