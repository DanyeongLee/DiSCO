for step in 1 3 5 10 15
do  
    python test.py +experiment=230709-fromtorsion-drugs-7steps-final gpu=0 dataset.torsion_test_path=../torsional-diffusion/geodiff_split_run/test_drugs_steps${step}.pkl train.test_starting_point=true
    python test.py +experiment=230709-fromtorsion-drugs-7steps-final gpu=0 dataset.torsion_test_path=../torsional-diffusion/geodiff_split_run/test_drugs_steps${step}.pkl
done