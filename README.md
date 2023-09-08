# NLP Review Classification - README

This repository contains code for classifying product reviews as positive or negative using Natural Language Processing (NLP) techniques.
 Below are instructions on how to use the provided `inference.py` script for making predictions and some considerations regarding the training Jupyter notebook.

## Inference Script (inference.py):

### How to Use:

To perform sentiment classification on a set of reviews and generate classification results, follow these steps:

1. **Prerequisites**:
   - Make sure you have Python 3.x installed.
   - Install the required Python packages by running: `pip install -r requirements.txt`

2. **Running the Inference Script**:
   - Open your terminal or command prompt.
   - Navigate to the project directory where `inference.py` is located.

3. **Command Syntax**:
   - Use the following command format to run the script:
     ```bash
     python3 inference.py input_reviews.csv output_predictions.csv
     ```
     - `input_reviews.csv`: The CSV file containing the reviews you want to classify. Make sure the file format matches the provided dataset.
     - `output_predictions.csv`: The output file where classification results will be saved.

4. **Output**:
   - The script will process the input reviews and generate classification results in the specified output file (`output_predictions.csv`).
   - The output file will contain columns for review IDs and predicted labels (1 for positive, 0 for negative).

### Note:

- You can use your own review data as long as it follows the same CSV format as the provided dataset.

## Training Jupyter Notebook:

### Considerations:

- We have included a Jupyter notebook (`train_model.ipynb`) that demonstrates the model training process. However, please keep the following considerations in mind:
  
  - **Training Time**: I would caution you against running any cells because they usually take a lot of time to run.

- If you intend to run the training notebook, consider the following tips:

  - **Training Data Size**: Experiment with smaller subsets of data initially for faster iterations.

  - **Google Colab**: Consider using Google Colab for training if you have access to it. It provides free GPU resources, which can significantly speed up model training.

