#load required packages
import numpy as np
import pandas as pd
import os
import re
import random
import tensorflow as tf
import keras
from flask import Flask, render_template, url_for, request
from keras.models import load_model
import h5py
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.framework import ops
ops.reset_default_graph()
global graph, model
#graph = tf.get_default_graph()


cwd = os.getcwd()
cust_df = pd.read_csv("data/profilesniche_prepared.csv")
model = load_model( "models/my_model.h5")
perfumes_final_data = pd.read_csv("data/perfumes_final_data.csv")
print("data loaded")
cust_df = cust_df[['IDcustomer', 'text']]
cust_df = cust_df[cust_df['text'].isna()==False]
perfume_info = perfumes_final_data[['ID_perfume', 'title']]
perfumes_final_data = perfumes_final_data.drop(['ID_perfume', 'title'], axis=1)
perumes_old = pd.read_csv("data/perfumes_old.csv")
products_nodes = pd.read_csv("data/products_for_nodes.csv")

loop_range = [str(i) for i in range(0, 66)]
#
#removeing names infromt of nodes

vals = ['Top0', 'Top1', 'Top2', 'Top3', 'Middle0', 'Middle1', 'Middle2', 'Base0', 'Base1']
for i in ['0', '1', '2', '3']:
    print(i)
    for val in vals:
        perumes_old[i] = perumes_old[i].apply(lambda x: x.replace(val,''))

#_______________________________________#

def clean_text(x):
    x = x.lower()
    x = re.sub('[^A-Za-z0-9]+', ' ', x)
    return x

cust_df['text'] = cust_df['text'].apply(lambda x:clean_text(x))

train_X = cust_df['text'].fillna("##").values

max_features=34791
maxlen = 300
embed_size=100
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
max_products = len(perumes_old)

def tokenize_and_return(x):
    x = tokenizer.texts_to_sequences([x])
    x = pad_sequences(x, maxlen=maxlen)
    print(x.shape)
    x = np.ones((max_products, 300)) * x
    return x

def get_tops(df):
    top_i = ','.join(word for word in df)
    top_i = top_i.split(",")
    top_i = set(top_i)
    return top_i

def get_nodes(my_list):
    top_list = []
    middle_list = []
    base_list = []

    for i in my_list:
        if type(i) == type(''):
            # print(i)
            if "Top" in i:
                val = i.replace("Top", "")
                val = ''.join([i for i in val if not i.isdigit()])
                top_list.append(val)
            elif "Middle" in i:
                val = i.replace("Middle", "")
                val = ''.join([i for i in val if not i.isdigit()])
                middle_list.append(val)
            elif "Base" in i:
                val = i.replace("Base", "")
                val = ''.join([i for i in val if not i.isdigit()])
                base_list.append(val)
    return top_list, middle_list, base_list

def get_predictions(input_text):
    input_text = tokenize_and_return(input_text)
    my_randoms = [random.randrange(1, 41905, 1) for _ in range(max_products)]
    my_array = np.array( my_randoms )
    my_array = np.array(range(0, len(perumes_old)))
    x_test_new = perfumes_final_data.iloc[my_array]
    print(input_text.shape)
    print(x_test_new.shape)
    ops.reset_default_graph()
    y_pred = model.predict([input_text, x_test_new])

    #with graph.as_default():
    #    y_pred = model.predict([input_text, x_test_new])
    print(y_pred.shape)

    some_df = pd.DataFrame(y_pred, columns=['prediction_0_val', 'prediction_1_val', 'prediction_2_val'])

    some_df['ID_perfume'] = perfume_info['ID_perfume'].iloc[my_array].values
    some_df['title'] = perfume_info['title'].loc[my_array].values
    some_df['accords'] = perumes_old['accords'].loc[my_array].values

    print(some_df.shape)
    for i in loop_range:
        some_df[i] = products_nodes[i].loc[my_array].values

    print(some_df.shape)
    del some_df['prediction_0_val']
    del some_df['prediction_1_val']


    result = some_df.sort_values(by='prediction_2_val', ascending=False)[:3]

    dummy = [i for i in result['accords']]
    total_accords = ','.join(word for word in dummy)
    words = total_accords.split(",")
    words_set = set(words)
    try:
        words_set.remove('unknown')
    except:
        pass

    final_top_layers = []
    final_middle_layers = []
    final_base_layers = []
    for i in range(0,3):
        my_list = result.iloc[i, 4:].values
        my_list = [i for i in my_list]
        top_list,middle_list, base_list = get_nodes(my_list)
        final_top_layers.append(top_list)
        final_middle_layers.append(middle_list)
        final_base_layers.append(base_list)


    return final_top_layers,final_middle_layers, final_base_layers, words_set

#final_top_layers,final_middle_layers, final_base_layers, words_set = get_predictions("i am interested in something rose and gold, i always like that the perfume should be good")

#print("helloooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
#print(final_top_layers)
#print(final_middle_layers)
#print(final_base_layers)
#print(words_set)
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictsomething', methods=['POST'])
def predictsomething():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        #print(data)
        str1 = ''.join(data)
        print(type(str1))
        final_top_layers, final_middle_layers, final_base_layers, words_set = get_predictions(str1)

        my_prediction = 1
    return render_template('home.html', prediction=my_prediction, message_data=str1,
                           final_top_layers = final_top_layers,
                           final_middle_layers=final_middle_layers,
                           final_base_layers = final_base_layers,
                           words_set=words_set)

if __name__ == '__main__':
    app.run(debug=True)