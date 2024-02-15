## Breast Cancer Detection with Keras and TensorFlow

This project implements a neural network model for breast cancer detection using Keras and TensorFlow libraries. This README file provides information about the project, installation instructions, and usage details.

### Project Overview

This project trains a neural network model to classify breast cancer using a dataset ( [Breast Cancer Wisconsin DataSet](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) ). The model uses Keras and TensorFlow libraries for building and training the network.

### Prerequisites

- Python 3.6 or higher
- TensorFlow 2.x ([https://www.tensorflow.org/](https://www.tensorflow.org/))
- Keras ([https://keras.io/](https://keras.io/))
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/abhay2132/breast-cancer.git
   ```

2. Navigate to the project directory:

   ```bash
   cd breast-cancer
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   This command will install all the necessary libraries listed in the `requirements.txt` file.

### Running the Project

1. Run the `main` script:

   ```bash
   python src/app.py
   ```

   This command will train the model on the available dataset and divide dataset into training and test set, and generate a confusion matrix.