import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemma = WordNetLemmatizer()
base_data= json.loads(open('base_data.json').read())

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=load_model('AI_chatbot.h5')

def clean_up_sent(sent):
    sent_words=nltk.word_tokenize(sent)
    sent_words=[lemma.lemmatize(word) for word in sent_words]
    return sent_words

def bag_of_words(sent):
    sentence_word= clean_up_sent(sent)
    bag=[0]*len(words)
    for w in sentence_word:
        for i, word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow= bag_of_words(sentence)
    res=model.predict(np.array([bow]))[0]
    error_threshold= 0.25
    result= [[i,r] for i,r in enumerate(res) if  r> error_threshold]

    result.sort(key=lambda x: x[1], reverse= True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]],"probability":str(r[1])})
    return return_list

def get_response(intent_list, intent_json):
    tag=intent_list[0]["intent"]
    list_of_intents= intent_json["base_data"]
    for i in list_of_intents:
        if i['tag']==tag:
            result = random.choice(i["responses"])
            break
    return result

print(" bot is running Now...")

while True:
    message= input("")
    ints= predict_class(message)
    res=get_response(ints,base_data)
    print(res)