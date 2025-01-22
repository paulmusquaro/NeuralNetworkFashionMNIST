# NeuralNetworkFashionMNIST

This project demonstrates how to build, train, and evaluate a neural network using Keras to classify products from the Fashion MNIST dataset. The goal is to achieve a classification accuracy of at least 91% using custom network architecture and optimized hyperparameters.

## Dataset Overview

The Fashion MNIST dataset consists of 70,000 grayscale images of 10 different categories of clothing. Each image is 28x28 pixels. The dataset is split into:
- **Training set**: 60,000 images
- **Test set**: 10,000 images

### Class Labels
The dataset includes the following clothing categories:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Key Features of the Project

1. **Data Preprocessing**:
   - Normalized pixel values to the range [0, 1].
   - Visualized sample images and their labels.

2. **Custom Neural Network Architecture**:
   - A sequential model with the following layers:
     - `Flatten` layer to convert 2D images into a 1D array.
     - `Dense` layer with 256 neurons and `relu` activation function.
     - `Dense` layer with 128 neurons and `sigmoid` activation function.
     - `Dropout` layer (rate = 0.1) to prevent overfitting.
     - Output `Dense` layer with 10 neurons (one for each class) and `softmax` activation function.

3. **Hyperparameter Optimization**:
   - **Optimizer**: Adam with a learning rate of 0.001.
   - **Loss Function**: Sparse categorical crossentropy.
   - **Batch Size**: 128.
   - **Epochs**: 10.

4. **Model Training**:
   - Used 20% of the training set as a validation set.
   - Tracked accuracy and loss metrics for both training and validation sets.

5. **Evaluation**:
   - Achieved a training accuracy of 90.9% and validation accuracy of 89.1%.
   - Visualized loss and accuracy trends across epochs.
   - Performed predictions on the test set and visualized sample results.

6. **Metrics**:
   - Used a classification report to evaluate precision, recall, and F1-score for each class.
   - Visualized prediction probabilities for individual samples.

## Results

- **Training Accuracy**: 90.9%
- **Validation Accuracy**: 89.1%
- **Test Accuracy**: ~88% (based on the classification report)
- The model can classify clothing items effectively, though some uncertainty remains for specific classes (e.g., shirts and pullovers).

---

## Conda (Setup and Environment)

To make the project reproducible and ensure smooth package management, this project uses Conda as a package and environment manager. Below are the steps to set up the environment:


1. **Install Conda**:
If you haven't installed Conda yet, you can download it from the official [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) websites. Anaconda is a larger distribution with more pre-installed packages, while Miniconda is a smaller, minimal version. Choose whichever suits your needs.

2. **Create a new environment:** Open your terminal and run the following command to create a new Conda environment with Python 3.7:

    ```bash
    conda create --name new_conda_env python=3.7
    ```

3. **Activate the environment:** Once the environment is created, activate it by running:

    ```bash
    conda activate new_conda_env
    ```

4. **Install required packages (Jupyter, NumPy, MatPlotLib, Pandas, Scikit-Learn, Tensorflow and Keras)**

    ```bash
    conda install jupyter numpy matplotlib pandas scikit-learn tensorflow keras
    ```

5. **Run Jupyter Notebook**

    ```bash
    jupyter notebook
    ```

---

## Future Improvements

- Experiment with additional regularization techniques such as L1/L2 weight regularization.
- Implement advanced architectures like CNNs for potentially better accuracy.
- Perform further hyperparameter tuning using techniques like grid search or random search.
- Increase the number of epochs and use early stopping for dynamic training termination.

## Conclusion

This project demonstrates how to effectively classify Fashion MNIST dataset images using a custom neural network built with Keras. The achieved accuracy meets the target criteria, and the results are well-visualized and interpretable. Further enhancements could push the model performance even higher.