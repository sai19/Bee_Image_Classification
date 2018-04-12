from __future__ import division
import pandas as pd
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.layers import multiply
from sklearn.metrics import roc_auc_score
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm 

train = pd.read_csv("train.csv");
x = {i:[] for i in range(40)}
for i in tqdm(range(2000)):
	for j in range(40):
		tr = train[(train["i"]==i)&(train["j"]==j)]
		if len(tr)>2:
			data = list(tr["t"])
			for k in range(len(data)-1):
				x[j].append(data[k+1]-data[k])

avg_gap = np.array([np.mean(x[i]) for i in range(40)])
pr_def_prices = np.zeros(40)
pr_ch_price = np.zeros((49,40))
for i in range(40):
	tr = train[(train["j"]==i)&(train["advertised"]==0)]
	value = tr["price"].unique()
	pr_def_prices[i] = value[0]

for t in range(49):
	p = np.zeros(40)
	for pr in list(train[(train["t"]==i)]["j"].unique()):
		p[pr] = train[(train["t"]==i)&(train["j"]==pr)]["price"].unique()[0]
	p[np.where(p)==0] = pr_def_prices[np.where(p)==0]
	pr_ch_price[t] = p  	
		
pr_buying_history = np.zeros((2000,49,40))
pr_weekly_data = np.zeros((2000,49,40))
pr_price_diff = np.zeros((2000,49,40))
for i in tqdm(range(2000)):
	a_count = -100*np.ones(40)
	for t in range(49):
		a = np.zeros(40)
		tr = train[(train["i"]==i)&(train["t"]==t)]	
		if len(tr)>0:
			bought_till = np.where(a_count>0)[0]
			update_history = [w for w in bought_till if w not in list(tr["j"])]
			a[np.array(list(tr["j"]))] = 1
			a_count[np.array(list(tr["j"]))] = 1
			if len(update_history)>0:
				a_count[np.array(update_history)] += 1
		else:
			a_count[np.where(a_count)>0] += 1
		pr_buying_history[i,t,:] = np.exp(a_count - avg_gap)
		pr_weekly_data[i,t,:] = a
		pr_price_diff[i,t,:] = pr_def_prices - pr_ch_price[t]

x_train_1,x_train_2,x_train_3,y_train = [],[],[],[]
x_test_1,x_test_2,x_test_3,y_test = [],[],[],[]  

for i in tqdm(range(2000)):
	for j in range(40-n_prev):
		if j<40-n_prev-1:
			x_train_1.append(pr_weekly_data[i,j:j+n_prev-1,:])
			x_train_2.append(pr_price_diff[i,j+1:j+n_prev,:])
			x_train_3.append(pr_buying_history[i,j+n_prev-1,:])
			y_train.append(pr_weekly_data[i,j+n_prev,:])
		else:
			x_test_1.append(pr_weekly_data[i,j:j+n_prev-1,:])
			x_test_2.append(pr_price_diff[i,j+1:j+n_prev,:])
			x_test_3.append(pr_buying_history[i,j+n_prev-1,:])
			y_test.append(pr_weekly_data[i,j+n_prev,:])	

x_train_1,x_train_2,x_train_3,y_train = np.array(x_train_1),np.array(x_train_2),np.array(x_train_3),np.array(y_train)
x_test_1,x_test_2,x_test_3,y_test = np.array(x_test_1),np.array(x_test_2),np.array(x_test_3),np.array(y_test)
x_train_1 = x_train_1.reshape((x_train_1.shape[0],x_train_1.shape[2],x_train_1.shape[1]))
x_train_2 = x_train_2.reshape((x_train_2.shape[0],x_train_2.shape[2],x_train_2.shape[1]))
x_test_1 = x_test_1.reshape((x_test_1.shape[0],x_test_1.shape[2],x_test_1.shape[1]))
x_test_2 = x_test_2.reshape((x_test_2.shape[0],x_test_2.shape[2],x_test_2.shape[1]))
#y_train = to_categorical(y_train,2)
#y_test = to_categorical(y_test,2)
input_1 = Input(shape=(40,n_prev-1))
out_1 = LSTM(16,activation="relu")(input_1)
out_1 = Dense(40,activation="relu")(out_1)

input_2 = Input(shape=(40,n_prev-1))
out_2 = LSTM(16,activation="relu")(input_2)
out_2 = Dense(40,activation="relu")(out_2)

input_3 = Input(shape=(40,))
out_3 = Dense(40,activation="relu")(input_3)

x = multiply([out_1,out_2,out_3])

out = Dense(64,activation="relu")(x)
out = Dense(40,activation="sigmoid")(out)
model = Model([input_1,input_2,input_3],out)

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
for i in range(10):
	model.fit([x_train_1,x_train_2,x_train_3],y_train,epochs=1,batch_size=32,validation_data=([x_test_1,x_test_2,x_test_3],y_test))
	predicted = model.predict([x_test_1,x_test_2,x_test_3])
	print(roc_auc_score(y_test.reshape((-1,1)),predicted.reshape((-1,1))))

