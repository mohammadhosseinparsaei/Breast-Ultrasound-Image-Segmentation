<div class="container">
    <div class="title-container">
        <h1>Breast Ultrasound Image Segmentation</h1>
    </div>
    <div class="image-container">
        <img src="https://github.com/mohammadhosseinparsaei/Breast-Ultrasound-Image-Segmentation/blob/main/sample%20image.png" alt="Blood Cell Sample" width="400" align="right">
    </div>
</div>

This project focuses on segmenting breast lesions in ultrasound images.
### Approach
For training and testing the image segmentation model, a U-Net neural network architecture has been employed using the Keras library.
### Dataset
In this project, two categories of image data have been utilized:

- Benign: Contains breast ultrasound images with benign lesions.
- Malignant: Contains breast ultrasound images with malignant or cancerous lesions.

### Data Access
The data is accessible and downloadable from [here](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).

- Citation: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. [DOI: 10.1016/j.dib.2019.104863]( https://doi.org/10.1016/j.dib.2019.104863)

### Tools and Libraries Used
- Python 3.6.15
- numpy 1.19.2
- pandas 1.1.5
- opencv 3.4.2
- scikit-learn 0.24.2
- keras 2.3.1
- tqdm 4.64.1
- matplotlib 3.3.4

### Model Evaluation with Random Images
<img src="https://github.com/mohammadhosseinparsaei/Breast-Ultrasound-Image-Segmentation/blob/main/evaluation.png" alt="images" width="600"/>

### IoU & Dice coefficient Plot
<img src="https://github.com/mohammadhosseinparsaei/Breast-Ultrasound-Image-Segmentation/blob/main/iou_dice_plot.png" alt="IoU & Dice plot" width="600"/>

### Model Architecture Plot
<img src="https://github.com/mohammadhosseinparsaei/Breast-Ultrasound-Image-Segmentation/blob/main/model_architecture_plot.png" alt="Architecture" width="400"/>
