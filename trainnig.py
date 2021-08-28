import json
from os import read
import numpy as np
import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.engine import training 

lemma= WordNetLemmatizer()

base_data= json.loads(open('base_data.json').read())

words=[]
classes=[]
documents=[]
ignore_letters=[',','!','?','.']

for data in base_data["base_data"]:
    for pattern in data["patterns"]:
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,data["tag"]))
        if data["tag"] not in classes:
            classes.append(data["tag"])

words= [lemma.lemmatize(word) for word in words if word not in ignore_letters]
words= sorted(set(words))
classes= sorted(set(classes))

pickle.dump(words,open("words.pkl","wb"))
pickle.dump(classes,open("classes.pkl","wb"))


training=[]
output_empty=[0]*len(classes)

for document in documents:
    bag=[]
    word_pattern= document[0]
    word_pattern= [lemma.lemmatize(word.lower()) for word in word_pattern]
    for word in words:
        bag.append(1) if word in word_pattern else bag.append(0)

    output_row=list(output_empty)
    output_row[classes.index(document[1])] =  1
    training.append([bag,output_row])


random.shuffle(training)
training=np.array(training)

x_train=list(training[:,0])
y_train=list(training[:,1])

model=Sequential()
model.add(Dense(128,input_shape=(len(x_train[0]),),activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation="softmax"))

sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)

model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])
ver=model.fit(np.array(x_train),np.array(y_train), epochs=200,batch_size=5)

model.save('AI_chatbot.h5',ver)

print("done")