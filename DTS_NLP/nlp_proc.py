import pickle
from difflib import SequenceMatcher
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def main_proc(inp):
  with open("fit.txt", "rb") as fp:
    b = pickle.load(fp)

  sent = []
  f = open("nlp_dict.txt", "r")
  for i in f:
    sent.append(i.strip())

  max = 0.0
  max_id = 0
  for i in sent:
    temp = SequenceMatcher(None, inp, i).ratio()
    if temp > max:
      max = round(temp,7)
      max_id = i

  new_model = tf.keras.models.load_model('cnn_model.h5')
  tokenizer = Tokenizer(num_words=26)
  tokenizer.fit_on_texts(b)
  ac_1 = list(inp)
  ac_2 = tokenizer.texts_to_sequences([ac_1])
  ac_3 = pad_sequences(ac_2, padding='post', maxlen=13)
  lbl = new_model.predict(ac_3)

  acc = ''
  acc_conf = 0
  if lbl[0, 0] > lbl[0, 1] and lbl[0, 0] > lbl[0, 2]:
    acc = 'Chinese'
    acc_conf = lbl[0, 0]
  elif lbl[0, 1] > lbl[0, 0] and lbl[0, 1] > lbl[0, 2]:
    acc = 'Indian'
    acc_conf = lbl[0, 1]
  else:
    acc = 'Malay'
    acc_conf = lbl[0, 2]

  acc_conf = round(float(acc_conf),7)
  print("Input string:", inp, "\nMatched:", max_id, "with similarity of", max, "\nAccent:", acc, "with confidence of",
        acc_conf)
  return max_id, max, acc, acc_conf