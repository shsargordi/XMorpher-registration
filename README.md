# XMorpher deformable registration between cardiac CT scans. CT pairs must undergo affine registration before deformable registration.



# Create environment
conda env create -f environment.yml

# Activate
conda activate XMorpher-Reg

# Search for bin anywhere in your environment 
find /path/to/miniconda3/envs/XMorpher-Reg/bin 


# Data organization
The data organization example is shown below: 
![alt text](image.png)

The folders train_labeled_unlabeled and train_unlabeled_unlabeled are used for training, and the data in the test folder is used for testing.

To prepare the dataset, follow the instructions in Prepare-Data/preprocessing.ipynb.

# Run the script
Run python Uns_uptrain.py for unsupervised training.
Run python Semi_train.py for semi-supervised training.


# XMorpher-registration
