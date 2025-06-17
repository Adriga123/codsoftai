import os

import string
import numpy as np
from tqdm import tqdm
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.models import load_model
import pickle

import os
import string
import numpy as np
from tqdm import tqdm
import pickle



dataset_path = 'Flickr8k_Dataset'
captions_path = 'Flickr8k.token.txt'


def load_captions(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    descriptions = {}
    for line in lines:
        tokens = line.strip().split('\t')
        image_id, caption = tokens[0].split('#')[0], tokens[1]
        caption = 'startseq ' + caption.lower().translate(str.maketrans('', '', string.punctuation)) + ' endseq'
        if image_id not in descriptions:
            descriptions[image_id] = []
        descriptions[image_id].append(caption)
    return descriptions

descriptions = load_captions(captions_path)


def extract_features(directory):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    features = {}
    for name in tqdm(os.listdir(directory)):
        filename = os.path.join(directory, name)
        img = image.load_img(filename, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = model.predict(img, verbose=0)
        features[name] = feature
    return features

features = extract_features(dataset_path)

with open("features.pkl", "wb") as f:
    pickle.dump(features, f)


all_captions = []
for captions in descriptions.values():
    all_captions.extend(captions)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in all_captions)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)


def data_generator(descriptions, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key, captions in descriptions.items():
            n += 1
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                yield [[np.array(X1), np.array(X2)], np.array(y)]
                X1, X2, y = [], [], []
                n = 0

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = define_model(vocab_size, max_length)

generator = data_generator(descriptions, features, tokenizer, max_length, vocab_size, batch_size=32)
model.fit(generator, epochs=10, steps_per_epoch=100, verbose=1)

model.save("caption_model.h5")

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence_ = tokenizer.texts_to_sequences([in_text])[0]
        sequence_ = pad_sequences([sequence_], maxlen=max_length)
        yhat = model.predict([photo, sequence_], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '')

img_path = os.path.join(dataset_path, '667626_18933d713e.jpg')
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
feature = vgg_model.predict(img, verbose=0)

caption = generate_caption(model, tokenizer, feature, max_length)
print("Generated Caption:", caption)
