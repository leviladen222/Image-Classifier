# Image-Classifier

This project trains and evaluates an image classifier in the notebook `image-classification.ipynb` using PyTorch and torchvision (ResNet-50 transfer learning).

## Notebook overview

The notebook is organized into these stages:

- Imports and environment setup
- Custom dataset class (`ImageDataset`) for loading labeled training images
- Image augmentation and train/test split generation
- Training utilities (`training_loop`, optional `fine_tune_model`)
- Model definition and training run (ResNet-50 with a custom classification head)
- Prediction pipeline for test images and CSV export

## Dataset layout expected by the notebook

The notebook expects the following folder structure:

- `data/train/train/<class_id>/<image_name>.jpg`
  - Class folders are numeric labels (0-99)
- `data/test/test/<image_name>.jpg`
  - Test images are named like `0.jpg`, `1.jpg`, ..., `999.jpg`

## How training works in `image-classification.ipynb`

- Uses transfer learning with `torchvision.models.resnet50(weights='IMAGENET1K_V2')`
- Replaces the final fully connected layer with a custom head (`ModelHead`) for 100 classes
- Applies random augmentation on training samples
- Tracks train/test loss and accuracy each epoch
- Saves the best model checkpoint as `<model_name>-<test_accuracy>.pth`

Note: one recorded notebook run was interrupted (`KeyboardInterrupt`) during training, which is normal if execution was manually stopped.

## Run the notebook

1. Open `image-classification.ipynb`.
2. Run cells from top to bottom.
3. Wait for training to generate a `.pth` checkpoint.
4. In the prediction section, set `model_name` to the saved checkpoint filename.
5. Run prediction cells to generate `submission.csv`.

## Output files

- Model checkpoints: `*.pth` (saved during training when test accuracy improves)
- Predictions file: `submission.csv` with columns:
  - `ID` (image filename)
  - `Label` (predicted class)
