Source: https://github.com/ellisdg/3DUnetCNN, https://arxiv.org/abs/1606.06650

Designed a Keras model for 3D segmentation of volumetric data for detecting modalities in brain tumor https://www.med.upenn.edu/sbia/brats2018/registration.html

Assumption: Brats dataset is preprocessed and saved as data/preprocessed original
 
Steps:

- Create conda environment from env.yaml
- Cd VolumetricSegmentationGitlab
- Run python train_val_split.py: Creates data file (brats_data.h5) and splits data into training and validation ids (training_ids.pkl, validation_ids.pkl)
- Run python train_v1.ipynb or train_v2.ipynb: Creates tumor_model.h5 in model, and segmentation predictions in predictions folder for validation ids in validation_ids.pkl
