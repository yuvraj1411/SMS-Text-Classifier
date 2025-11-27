📧 SMS Spam Detection using Deep Learning
This Google Colab notebook provides a complete, self-contained implementation of an SMS Spam Detection model using a Sequential Neural Network built with TensorFlow and Keras. It employs Natural Language Processing (NLP) techniques to classify text messages as either ham (legitimate) or spam.

🔗 Colab Notebook Link: https://colab.research.google.com/drive/1Sj8DwIoZBtn1AH-1Zoyddc5RoTsLyE3N?usp=drive_link

🚀 Workflow & Key Components
The notebook automates the entire machine learning pipeline, making it easy to run and reproduce:

1. Setup: Installs necessary libraries (tf-nightly, tensorflow-datasets) and imports TensorFlow, Pandas, and NumPy.

2. Data Processing:

   Downloads and loads training and validation SMS datasets.
   Converts 'ham' / 'spam' labels to 0 / 1 integers.

3. Tokenization & Padding: Converts messages into numerical sequences and standardizes the length to 120 for model input.

4. Model Architecture: A deep learning model optimized for text classification:

5. Embedding Layer: Maps words to 64-dimensional vectors.

   Global Average Pooling 1D - Efficiently summarizes the message's features.
   Dense Layers (with Dropout) -  Performs classification, concluding with a Sigmoid activation to output the spam probability.

6. Training & Evaluation:

   The model is compiled with the Adam optimizer and binary cross-entropy loss.
   It is trained for 30 epochs with validation on the test data.

7. Prediction: A utility function, predict_message(), is included to classify new messages and a test suite runs automatically to confirm functionality.

💻 Usage
Open the Colab link and run all cells sequentially. No local setup or file configuration is required. The notebook handles all dependencies and execution steps automatically.

📜 License
This project is made available for educational purposes.
