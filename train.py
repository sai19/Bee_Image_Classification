from __future__ import division
import pandas as pd
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.layers import multiply
from sklearn.metrics import roc_auc_score
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm 
from keras import backend as K
import math
def norm(x, mean, sd):
    var = float(sd)**2
    pi = math.pi
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def categorical_crossentropy(y_true, y_pred):
	return K.categorical_crossentropy(y_true, y_pred,from_logits=True)

train = pd.read_csv("train.csv");
x = {i:[] for i in range(40)}
new_pr_buy_prob = np.zeros(2000)
for i in tqdm(range(2000)):
	count = 0
	new_count = 0
	a_count = -20*np.ones(40)
	for t in range(49):
		a = np.zeros(40)
		tr = train[(train["i"]==i)&(train["t"]==t)]	
		if len(tr)>0:
			count += 1 
			bought_till = np.where(a_count>0)[0]
			update_history = [w for w in bought_till if w not in list(tr["j"])]
			new_bought = [w for w in list(tr["j"]) if w not in bought_till]
			if len(new_bought)>0:
				new_count += 1	 
			a[np.array(list(tr["j"]))] = 1
			a_count[np.array(list(tr["j"]))] = 1
			if len(update_history)>0:
				a_count[np.array(update_history)] += 1
		else:
			a_count[np.where(a_count)>0] += 1
	new_pr_buy_prob[i] = new_count/count			
for i in tqdm(range(2000)):
	for j in range(40):
		tr = train[(train["i"]==i)&(train["j"]==j)]
		if len(tr)>1:
			data = list(tr["t"])
			for k in range(len(data)-1):
				x[j].append(data[k+1]-data[k])

product_proba = np.zeros((49,40))
prob_till = np.zeros((4,40))
for i in range(49):
	tr = train[train["t"]==i]
	out = dict(tr["j"].value_counts())
	out = [0 if n not in out.keys() else out[n] for n in range(40)]
	prob_till[i%4] = prob_till[i%4] + np.array(out)
	product_proba[i] = prob_till[i%4]/np.sum(prob_till[i%4]) 				

avg_gap = np.array([-20 if len(x[i])==0 else np.mean(x[i]) for i in range(40)])
std_gap = np.array([2 if len(x[i])==0 else np.std(x[i]) for i in range(40)])
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
	a_count = -20*np.ones(40)
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
		non_bought = len(np.where(a_count<0)[0])	
		p = np.array([new_pr_buy_prob[i]/non_bought if a_count[n]<0 else norm(a_count[n],avg_gap[n],std_gap[n]) for n in range(40)])	
		pr_buying_history[i,t,:] = p#np.multiply(p,product_proba[t])
		pr_weekly_data[i,t,:] = a
		pr_price_diff[i,t,:] = 1 - np.divide(pr_ch_price[t],pr_def_prices)
x_train,y_train = [],[]
x_test,y_test = [],[] 
n_prev = 5
for i in tqdm(range(2000)):
	for j in range(48):
		#if j>20:
			if j<48-1 :
				data = np.concatenate((pr_weekly_data[i,j,:],pr_price_diff[i,j+1,:],pr_buying_history[i,j,:],product_proba[j]))
				x_train.append(data)
				y_train.append(pr_weekly_data[i,j+1,:])
			else:
				data = np.concatenate((pr_weekly_data[i,j,:],pr_price_diff[i,j+1,:],pr_buying_history[i,j,:],product_proba[j]))
				x_test.append(data)
				y_test.append(pr_weekly_data[i,j+1,:])

x_train,y_train = np.array(x_train),np.array(y_train)
x_test,y_test = np.array(x_test),np.array(y_test)

input_1 = Input(shape=(x_train.shape[1],))
out_1 = Dense(64,activation="relu")(input_1)
out_1 = Dense(64,activation="relu")(out_1)
out = Dense(64,activation="relu")(out_1)
out = Dense(40,activation="sigmoid")(out)
model = Model(input_1,out)

model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=["accuracy"])
for i in range(100):
	model.fit(x_train,y_train,epochs=1,batch_size=32,validation_data=(x_test,y_test))
	predicted = model.predict(x_test)
	print(roc_auc_score(y_test.reshape((-1,1)),predicted.reshape((-1,1))))

