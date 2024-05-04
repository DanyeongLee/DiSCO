# DiSCO: Diffusion Schrödinger Bridge for Molecular Conformer Optimization

# Requirements
- Python==3.8
- CUDA==11.8

- We provide script (setting.sh) to install all the required packages.

# Dataset
- GEOM dataset, and pre-generated conformers for this dataset with three baseline methods (except for RDKit ETKDG) should be appropriately downloaded.

- All the data, pre-generated conformers with baseline methods can be downloaded from https://drive.google.com/drive/folders/1XgmgSMNpnb-XE15inieNnN0zKCr1xy0d?usp=sharing.
    - torsional-diffusion.tar.gz: pre-generated conformers with torsional diffusion
    - dmcg.tar.gz: pre-generated conformers with dmcg
    - rdkit-clustering.tar.gz: pre-generated conformers with rdkit+clustering
    - disco_data.tar.gz: preprocessed GEOM dataset

- Please unzip the all tar.gz files using tar -zxvf commands, and locate them accordingly:
    - dmcg, torsional-diffusion, rdkit-clustering should be in parent directory of this working directory.
        - ../dmcg
        - ../torsional-diffusion
        - ../rdkit-clustering
    - data, outputs should be in this working directory.
        - ./data
        - ./outputs (not needed for training from scratch)

# Running
- Major scripts to run our experiments can be found in scripts directory
    - run_train.sh: Train all four baseline methods for two datasets. This could take very long time.
    - run_test.sh: Test ensemble RMSD metrics of four baseline methods.
    - test_prop.sh: Run xTB to calculate ensemble property.
    - noise_performance.sh: Run DiSCO after the addition of Gaussion noise to the output of torsional diffusion and calculate the ensemble RMSD metrics.
    - steps_performance.sh: Run DiSCO to the output of torsional diffusion with varying diffusion steps.
    - mmff_iters_performance.sh: Run DiSCO after the application of MMFF with varying iterations to the output of torsional diffusion.
    - mmff_four_models_performance: Run DiSCO after the application of MMFF with 200 iterations to the output of four baseline methods.

- Other codes
    - local_structure_error.py: Compare the local structure prediction error of torsional diffusion and torsional diffusion + DiSCO.
    - traj_energy_xtb.py: Calculate the energy of intermediate samples from DiSCO with xTB.



