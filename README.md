# Iris Flower Classification (Neural Network with Regularization)

This project implements a neural network using TensorFlow and Keras to classify the famous Iris dataset into its three species (Setosa, Versicolor, Virginica) based on four features (sepal length/width, petal length/width). It incorporates regularization techniques like L2, Dropout, and Batch Normalization to improve model robustness and prevent overfitting.

## Project Overview

The notebook demonstrates a complete workflow for multi-class classification using tabular data:
* **Data Loading:** Uses the Iris dataset available in `sklearn.datasets`.
* **Preprocessing:**
    * Splits the data into training (80%) and testing (20%) sets.
    * **Standardizes** the features using `StandardScaler` to have zero mean and unit variance.
    * **One-hot encodes** the target labels (0, 1, 2) into a categorical format (e.g., `[1, 0, 0]`).
* **Model Building:** Defines a `Sequential` neural network with:
    * Dense layers (`relu` activation).
    * **L2 Regularization** (`kernel_regularizer=regularizers.l2(0.001)`) added to dense layers to penalize large weights.
    * **Batch Normalization** layers to stabilize and speed up training.
    * **Dropout** layers (`Dropout(0.3)`) to randomly set a fraction of input units to 0 during training, preventing over-reliance on specific neurons.
    * A final `Dense` layer with `softmax` activation for multi-class probability output.
* **Training:** Trains the model for 50 epochs using the Adam optimizer and `categorical_crossentropy` loss.
* **Evaluation:** Evaluates the model's performance on the unseen test set using accuracy and generates a detailed `classification_report` (precision, recall, F1-score).

---

## Dataset: Iris

* **Source:** `sklearn.datasets.load_iris()`
* **Samples:** 150
* **Features:** 4 (Sepal Length, Sepal Width, Petal Length, Petal Width) - all in cm.
* **Classes:** 3 (Iris Setosa, Iris Versicolor, Iris Virginica)


---

## Model Architecture

The `Sequential` model includes these layers:

1.  `Dense` (64 units, `relu`, L2 regularization, input_shape=(4,))
2.  `BatchNormalization`
3.  `Dropout` (0.3 rate)
4.  `Dense` (64 units, `relu`, L2 regularization)
5.  `BatchNormalization`
6.  `Dropout` (0.3 rate)
7.  `Dense` (3 units, `softmax` activation) - Output layer for 3 classes.

**Compilation:**
* **Optimizer:** `adam`
* **Loss:** `categorical_crossentropy`
* **Metrics:** `accuracy`

---

## Requirements

You'll need the following Python libraries:

* `tensorflow`
* `scikit-learn`
* `numpy`
* `matplotlib`

Install them using pip:
```bash
pip install tensorflow scikit-learn numpy matplotlib
