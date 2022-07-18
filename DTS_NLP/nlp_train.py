import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
import pickle

accent = pd.read_excel("NLP Data - finalv2.xlsx")

temp = []
max = 0 
for i in accent["word"]:
  leng = len(list(i))
  if leng > max:
    max = leng
  temp.append(list(i))

x = temp
y = accent.drop(labels = ['word'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1000)
with open("fit.txt", "wb") as fp:
  pickle.dump(x_train, fp)

tokenizer = Tokenizer(num_words=26)
tokenizer.fit_on_texts(x_train)
Xcnn_train = tokenizer.texts_to_sequences(x_train)
Xcnn_test = tokenizer.texts_to_sequences(x_test)
vocab_size = len(tokenizer.word_index) + 1

maxlen = max
Xcnn_train = pad_sequences(Xcnn_train, padding='post', maxlen=maxlen)
Xcnn_test = pad_sequences(Xcnn_test, padding='post', maxlen=maxlen)

embedding_dim = 200
textcnnmodel = Sequential()
textcnnmodel.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
textcnnmodel.add(layers.Conv1D(128, 5, activation='relu'))
textcnnmodel.add(layers.GlobalMaxPooling1D())
textcnnmodel.add(layers.Dense(10, activation='relu'))
textcnnmodel.add(layers.Dense(3, activation='sigmoid'))
textcnnmodel.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])
textcnnmodel.summary()

textcnnmodel.fit(Xcnn_train, y_train,
                     epochs=100,
                     verbose=False,
                     validation_data=(Xcnn_test, y_test),
                     batch_size=10)
loss, accuracy = textcnnmodel.evaluate(Xcnn_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = textcnnmodel.evaluate(Xcnn_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

textcnnmodel.save('cnn_model.h5')