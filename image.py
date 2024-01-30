import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
from PIL import Image


def extract_image_features(img_path):
    model = ResNet50(weights='imagenet')
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.reshape((-1,))


def load_captions(file_path):
    with open(file_path, 'r') as file:
        captions = file.readlines()
    captions = [caption.strip() for caption in captions]
    return captions


def tokenize_captions(captions):
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(captions)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    output_sequences = []

    for caption in captions:
        tokenized_caption = tokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(tokenized_caption)):
            input_sequences.append(extract_image_features(image_paths[0]))
            output_sequences.append(tokenized_caption[:i+1])

    max_sequence_length = max([len(seq) for seq in output_sequences])
    padded_output_sequences = pad_sequences(output_sequences, maxlen=max_sequence_length, padding='post')

    X = np.array(input_sequences)
    y = tf.keras.utils.to_categorical(padded_output_sequences, num_classes=total_words)

    return X, y, tokenizer, total_words, max_sequence_length


def build_model(input_shape, total_words, max_sequence_length):
    model = Sequential()
    model.add(Dense(256, input_dim=input_shape, activation='relu'))
    model.add(RepeatVector(max_sequence_length))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(total_words, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def generate_caption(model, tokenizer, image_path, max_sequence_length):
    input_sequence = np.array([extract_image_features(image_path)])
    caption = []

    for i in range(max_sequence_length):
        prediction = model.predict(input_sequence)
        predicted_word_index = np.argmax(prediction[0, i, :])
        predicted_word = tokenizer.index_word.get(predicted_word_index, "<OOV>")
        caption.append(predicted_word)

        if predicted_word == "<EOV>":
            break

        input_sequence = np.concatenate([input_sequence, np.array([[predicted_word_index]])], axis=1)

    return ' '.join(caption)


image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
caption_file_path = "path/to/captions.txt"


captions = load_captions(caption_file_path)
X, y, tokenizer, total_words, max_sequence_length = tokenize_captions(captions)


model = build_model(X.shape[1], total_words, max_sequence_length)
model.fit(X, y, epochs=50, verbose=1)


model.save("image_captioning_model.h5")


for img_path in image_paths:
    generated_caption = generate_caption(model, tokenizer, img_path, max_sequence_length)
    print(f"Generated Caption for {os.path.basename(img_path)}:", generated_caption)