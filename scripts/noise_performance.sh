for noise in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do  
    python test.py +experiment=230709-fromtorsion-drugs-7steps-final gpu=0 dataset.etkdg_noise=${noise} train.test_starting_point=true
    python test.py +experiment=230709-fromtorsion-drugs-7steps-final gpu=0 dataset.etkdg_noise=${noise}
done