python gen_prop.py +experiment=230709-fromtorsion-qm9-7steps-final gpu=0
python xtb_prop.py --file outputs/230709-fromtorsion-qm9-7steps-final/gen_samples_prop.pkl

python gen_prop.py +experiment=230803-fromrdkit-qm9-10steps-final gpu=0
python xtb_prop.py --file outputs/230803-fromrdkit-qm9-10steps-final/gen_samples_prop.pkl