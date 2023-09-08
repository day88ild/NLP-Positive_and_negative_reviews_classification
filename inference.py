import pandas as pd
import argparse
import numpy as np
import tensorflow as tf
import json

# Define a function to perform inference
def perform_inference(input_file, output_file, model):
    # Read the input CSV file containing reviews
    input_data = pd.read_csv(input_file)


    # Load the JSON string from a file
    with open('models/tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)

    # Convert the JSON string back to a tokenizer
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    
    sequences = tokenizer.texts_to_sequences(list(input_data.text.values))
    
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=150, padding="post", truncating="post")

    # Perform classification using the trained model
    predictions = model.predict(sequences).flatten().round()  # Assuming 'review_text' is the column containing text data
    


    # Create a DataFrame to store the results
    results_df = pd.DataFrame({'id': input_data['id'], 'predicted_label': predictions})

    # Save the results to the output CSV file
    results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="NLP Review Classification Inference")
    parser.add_argument("input_file", type=str, help="Input CSV file with reviews")
    parser.add_argument("output_file", type=str, help="Output CSV file for classification results")

    args = parser.parse_args()

    # Load the trained model (you need to replace 'model.pkl' with the actual model file)
    trained_model = tf.keras.models.load_model("models/model_review_clf.h5")

    # Perform inference and save results
    perform_inference(args.input_file, args.output_file, trained_model)

