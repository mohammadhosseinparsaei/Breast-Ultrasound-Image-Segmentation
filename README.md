# Breast Ultrasound Image Segmentation
This project focuses on detecting and segmenting breast lesions using breast ultrasound image data.
### Approach
For training and testing the image segmentation model, a U-Net neural network architecture has been employed using the Keras library.
### Dataset
In this project, three categories of image data have been utilized:

- Normal: Contains breast ultrasound images without any lesions.
- Benign: Contains breast ultrasound images with benign lesions.
- Malignant: Contains breast ultrasound images with malignant or cancerous lesions.
### Data Access
The data is accessible and downloadable from [here](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).

- Citation: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. [DOI: 10.1016/j.dib.2019.104863]( https://doi.org/10.1016/j.dib.2019.104863)

### Tools and Libraries Used
- Python 3.6.15
- numpy 1.19.2
- pandas 1.1.5
- scikit-learn 0.24.2
- keras 2.3.1
- tqdm 4.64.1
- matplotlib 3.3.4
### Model Evaluation with Random Images
![images](https://github.com/mohammadhosseinparsaei/Breast-Ultrasound-Image-Segmentation/blob/main/evaluation.png)
### Loading the Model and Evaluation with Selected Images
![selected](https://github.com/mohammadhosseinparsaei/Breast-Ultrasound-Image-Segmentation/blob/main/selected_images.png)
### Model Architecture Plot
![Architecture](https://github.com/mohammadhosseinparsaei/Breast-Ultrasound-Image-Segmentation/blob/main/model_architecture_plot.png)
### Accuracy & Loss Plot
![Accuracy & Loss plot](https://github.com/mohammadhosseinparsaei/Breast-Ultrasound-Image-Segmentation/blob/main/accuracy_loss_plot.png)
