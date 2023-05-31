
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
from sklearn.model_selection import train_test_split
import pandas as pd 
#import nltk
from tensorflow.keras.callbacks import ModelCheckpoint
import csv


### DATA

data_url = "mi_dataset.csv"

# READ DATA
def read_csv_data(file):
    tokenized = list()
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            text = row[1]
            tokenized.append(text)

    return tokenized

sentences = read_csv_data(data_url)


### DATA PREPARING
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
train_sequences = tokenizer.texts_to_sequences(sentences)
vocab_size = len(tokenizer.word_counts)
max_length = max([len(x) for x in train_sequences])

def generate_padded_sequences(input_sequences):
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_length, padding='pre'))
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=vocab_size)
    return predictors, label

x_full, y_full = generate_padded_sequences(train_sequences)

#SPLIT DATASET - 80% WE USE TO TRAIN MODEL, 20% TO TEST RESULTS
x_train, x_test = train_test_split(x_full, test_size=0.2, random_state=0)
y_train, y_test = train_test_split(y_full, test_size=0.2, random_state=0)



#MODEL SETTINGS
embedding_dim = 32
num_epochs = 30
learning_rate=0.01

#MODEL ARCHITECTURE
model_lstm = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model_lstm.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


#LOAD WEIGHTS/MODEL
model_lstm.load_weights("weights-improvement-30-3.5959.hdf5")

# Re-evaluate the model
loss, acc = model_lstm.evaluate(x_test, y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#model_lstm.summary()

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#TRAINING
'''

history_lstm = model_lstm.fit(x_train, y_train,
                    validation_data=(x_test, y_test), 
                    epochs=num_epochs, 
                    verbose=2,
                    callbacks=callbacks_list
                    )

'''



#GENERATION TEXT
def generate_text(seed_text, num_words, model, max_sequence_len):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        #predicted = model.predict_classes(token_list, verbose=0)
        predict_x=model.predict(token_list)
        predicted=np.argmax(predict_x,axis=1)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text

seed_text = "even if "
new_text = generate_text(seed_text, 10, model_lstm, max_length)
print(new_text)