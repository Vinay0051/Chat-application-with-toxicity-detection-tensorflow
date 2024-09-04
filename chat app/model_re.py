import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense
from matplotlib import pyplot as plt

class ToxicityAnalyzer:
    def __init__(self, max_features=200000, sequence_length=1800):
        self.max_features = max_features
        self.sequence_length = sequence_length
        self.vectorizer = TextVectorization(
            max_tokens=self.max_features,
            output_sequence_length=self.sequence_length,
            output_mode='int'
        )
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(Embedding(self.max_features + 1, 32))
        model.add(Bidirectional(LSTM(32, activation="tanh")))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(6, activation="sigmoid"))
        model.compile(loss="BinaryCrossentropy", optimizer='Adam')
        model=tf.keras.models.load_model(r"C:\Users\vicky\Desktop\SDP\chat app\toxicity.h5")
        return model

    def preprocess_data(self, x, y):
        self.vectorizer.adapt(x.values)
        vectorized_text = self.vectorizer(x.values)
        dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
        dataset = dataset.cache().shuffle(160000).batch(16).prefetch(8)
        train = dataset.take(int(len(dataset) * 0.7))
        val = dataset.skip(int(len(dataset) * 0.7)).take(int(len(dataset) * 0.2))
        test = dataset.skip(int(len(dataset) * 0.9)).take(int(len(dataset) * 0.1))
        return train, val, test


    def train_model(self, train_data, val_data, epochs=1):
        history = self.model.fit(train_data, epochs=epochs, validation_data=val_data)
        return history

    def predict_toxicity(self, input_text):
        input_text = self.vectorizer(input_text)
        res = self.model.predict(np.expand_dims(input_text, 0))
        return res.tolist()[0]

    def analyze_user_toxicity(self, users, convo_list):
        d = {}
        user_convo_count = {}
        for convo in convo_list:
            convo = convo.split()
            user = convo[0]
            convo = convo[2:]
            input_text = " ".join(convo)
            input_text = input_text.replace('\n', '')
            res = self.predict_toxicity(input_text)

            if user not in users:
                users.append(user)
                d[user] = [0, 0, 0, 0, 0, 0]
                user_convo_count[user] = 0

            user_convo_count[user] += 1
            result_list = [elem1 + elem2 for elem1, elem2 in zip(d[user], res)]
            d[user] = result_list

        return users, d, user_convo_count

    def visualize_user_toxicity(self, user,userno, avg_toxic, avg_severe_toxic, avg_obscene, avg_threat, avg_insult, avg_identity_hate):
        trait = 0
        if avg_toxic > 0.5:
            print(f"{user} convo is toxic")
            trait += 1
        if avg_severe_toxic > 0.5:
            print(f"{user} convo is severe_toxic")
            trait += 1
        if avg_obscene > 0.5:
            print(f"{user} convo is obscene")
            trait += 1
        if avg_threat > 0.5:
            print(f"{user} convo is threatening")
            trait += 1
        if avg_insult > 0.5:
            print(f"{user} convo is insulting")
            trait += 1
        if avg_identity_hate > 0.5:
            print(f"{user} convo posses identity hate")
            trait += 1

        if trait > 3:
            print("The user posses severe threat")

        data = [avg_toxic, avg_threat, avg_identity_hate, avg_insult, avg_obscene, avg_severe_toxic]
        categories = ['Toxic', 'Threat', 'Identity_hate', 'Insult', 'Obscene', 'Severe_toxic']
        plt.bar(categories, data, color='#86bf91')

        plt.title(f'{user}')
        plt.savefig(f'C:\\Users\\vicky\\Desktop\\SDP\\chat app\\chatapp\\static\\images\\user{userno}.png')

def main():
    df = pd.read_csv(r"C:\Users\vicky\Desktop\SDP\chat app\chatapp\train.csv")
    x = df["comment_text"]
    y = df[df.columns[2:]].values

    # Assuming 'convo_list' is a list of conversation strings
    file_object = open(r"C:\Users\vicky\Desktop\SDP\chat app\chatapp\conversation.txt", "r")
    convo_list = file_object.readlines()
    file_object.close()

    users = []
    analyzer = ToxicityAnalyzer()

    train_data, val_data, test_data = analyzer.preprocess_data(x, y)  # Pass y here
    #analyzer.train_model(train_data, val_data, epochs=1)

    users, d, user_convo_count = analyzer.analyze_user_toxicity(users, convo_list)

    for user in d:
        avg_toxic = d[user][0] / user_convo_count[user]
        avg_severe_toxic = d[user][1] / user_convo_count[user]
        avg_obscene = d[user][2] / user_convo_count[user]
        avg_threat = d[user][3] / user_convo_count[user]
        avg_insult = d[user][4] / user_convo_count[user]
        avg_identity_hate = d[user][5] / user_convo_count[user]
        userno_list=list(d.keys())
        userno=userno_list.index(user)

        analyzer.visualize_user_toxicity(user,userno, avg_toxic, avg_severe_toxic, avg_obscene, avg_threat, avg_insult,
                                         avg_identity_hate)
        
        file_object = open(r"C:\Users\vicky\Desktop\SDP\chat app\chatapp\conversation.txt", "w")
        file_object.close()

if __name__ == "__main__":
    main()
