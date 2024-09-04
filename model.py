import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from matplotlib import pyplot as plt

df=pd.read_csv("train.csv")
x=df["comment_text"]
y=df[df.columns[2:]].values

max_features=200000
vectorizer=TextVectorization(max_tokens=max_features,output_sequence_length=1800,output_mode='int')
vectorizer.adapt(x.values)
vectorized_text=vectorizer(x.values)

dataset=tf.data.Dataset.from_tensor_slices((vectorized_text,y))
dataset= dataset.cache()
dataset= dataset.shuffle(160000)
dataset= dataset.batch(16)
dataset= dataset.prefetch(8)

train=dataset.take(int(len(dataset)*.7))
val= dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test= dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

model=Sequential()
model.add(Embedding(max_features+1,32))
model.add(Bidirectional(LSTM(32, activation="tanh")))
model.add(Dense(128, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(6, activation="sigmoid"))

model.compile(loss="BinaryCrossentropy",optimizer='Adam')
#history=model.fit(train,epochs=1,validation_data=val)

file_object=open("conversation.txt","r")
convo_list=file_object.readlines()
users=[]
d={}
user_convo_count={}
for convo in convo_list:
    convo=convo.split()
    user=convo[0]
    convo=convo[2:]
    input_text=" ".join(convo)
    input_text=vectorizer(input_text)
    batch=test.as_numpy_iterator().next()
    res=model.predict(np.expand_dims(input_text,0))
    my_list=res.tolist()
    my_list=my_list[0]
    if user not in users:
        users.append(user)
        d[user]=[0,0,0,0,0,0]
        user_convo_count[user]=1
    
    user_convo_count[user]+=1
    result_list= [elem1 + elem2 for elem1, elem2 in zip(d[user], my_list)]
    d[user]=result_list

print(d)
print(user_convo_count)
print(user)

#To find the average toxicity per user
trait=0
for user in d:
    avg_toxic=d[user][0]/user_convo_count[user]
    avg_severe_toxic=d[user][1]/user_convo_count[user]
    avg_obscene=d[user][2]/user_convo_count[user]
    avg_threat=d[user][3]/user_convo_count[user]
    avg_insult=d[user][4]/user_convo_count[user]
    avg_identity_hate=d[user][5]/user_convo_count[user]

    trait=0
    if avg_toxic>0.5:
        print(f"{user} convo is toxic")
        trait+=1
    if avg_severe_toxic>0.5:
        print(f"{user} convo is severe_toxic")
        trait+=1
    if avg_obscene>0.5:
        print(f"{user} convo is obscene")
        trait+=1
    if avg_threat>0.5:
        print(f"{user} convo is threatening")
        trait+=1
    if avg_insult>0.5:
        print(f"{user} convo is insulting")
        trait+=1
    if avg_identity_hate>0.5:
        print(f"{user} convo posses identity hate")
        trait+=1

    if trait>3:
        print("The user posses severe threat")

file_object.close()
file_object=open("conversation.txt","w")
file_object.close()