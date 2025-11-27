import sys
print(f"--- Starting in Python {sys.version} ---")
print("--- Uninstalling existing TensorFlow... ---")
!pip uninstall -y tensorflow tensorflow-cpu tensorflow-gpu tf-nightly tensorboard-plugin-profile
print("--- Installing compatible TensorFlow (tf-nightly)... ---")
!pip install -q tf-nightly

print("--- Importing libraries... ---")
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install -q tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(f"\n--- SUCCESS! ---")
print(f"TensorFlow version: {tf.__version__}")
print("Ready to run Cell 2.")

!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"


train_df = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_df = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])


train_df['label'] = train_df['label'].map({'ham': 0, 'spam': 1})
test_df['label'] = test_df['label'].map({'ham': 0, 'spam': 1})


train_data = train_df['message'].values
train_labels = train_df['label'].values

test_data = test_df['message'].values
test_labels = test_df['label'].values

print("Data loaded and prepared.")
print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 64
maxlen = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_data)

train_sequences = tokenizer.texts_to_sequences(train_data)
train_padded = pad_sequences(train_sequences, maxlen=maxlen, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(test_sequences, maxlen=maxlen, padding=padding_type, truncating=trunc_type)

train_labels_final = np.array(train_labels)
test_labels_final = np.array(test_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

num_epochs = 30
history = model.fit(train_padded,
                    train_labels_final,
                    epochs=num_epochs,
                    validation_data=(test_padded, test_labels_final),
                    verbose=2,
                   )

print("\nModel training complete.")

def predict_message(pred_text):
  new_sequence = tokenizer.texts_to_sequences([pred_text])
  new_padded = pad_sequences(new_sequence, maxlen=maxlen, padding=padding_type, truncating=trunc_type)


  prediction = model.predict(new_padded)[0][0]

  if prediction > 0.5:
    label = "spam"
  else:
    label = "ham"

  return [prediction, label]

print("Predict_message function is defined and ready.")

def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                   ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  print("--- RUNNING PREDICTION TEST ---")

  for i in range(len(test_messages)):
    prediction = predict_message(test_messages[i])
    pred_label = prediction[1]
    pred_prob = prediction[0]
    correct_label = test_answers[i]

    if (pred_label != correct_label):
      passed = False
      print(f"FAILED ON: '{test_messages[i]}'")
      print(f"  Expected: '{correct_label}', Got: '{pred_label}' (Prob: {pred_prob:.4f})\n")

  print("--- TEST COMPLETE ---")
  if (passed):
    print("You passed the challenge!")
  else:
    print("You failed the challenge.")

# Run the new test function
test_predictions()
