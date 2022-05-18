import sys
import csv
import math
import time
import random
import numpy as np
import pandas as pd
import plotly.express as px
from ast import literal_eval
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import numpy as np
from neupy import algorithms
from neupy.layers import *

from google_drive_downloader import GoogleDriveDownloader as gdd


def insert_next_s(case):
  case['next_state'] = case['state'].shift(-1)
  case = case.fillna('END')
  return case

def get_y(x, n, q, gamma):#, scale):
  q_next_s = 0
  if x['next_state'] != 'END':
    q_next_s = q[q['state']==x['next_state']].groupby('action')[n-1].mean().min()
  return x['cost'] + gamma * q_next_s

def train(model, x_train, q, max_it = MAX_IT, max_diff = MAX_DIFF, gamma = GAMMA):#, scale = False):
  start_time = time.time()
  for n in range(1,max_it):
    print(n)
    y = q.apply(lambda x: get_y(x, n, q, gamma), axis=1)
    model = model.fit(x_train, y)
    #model.fit(x_train, y)
    q[n] = model.predict(x_train)
    if (q[n-1] - q[n]).sum() < max_diff or (time.time()-start_time)>MAX_TIME: 
      break
  return q, n, time.time()-start_time

def init_q(h):
  aux = h.groupby('case').apply(insert_next_s)
  aux[0] = h.cost.max()
  return aux

def save_exec(alg, alg_espec, model_type, model_espec, gamma, it, train_time, q):
  #save metadata
  exec_metadata = [alg, alg_espec, model_type, model_espec, gamma, it, train_time]
  with open(PATH_RESULTS + 'metadata.csv','a') as f:
    writer = csv.writer(f)
    writer.writerow(exec_metadata)
  #save q
  q_filename = ('-').join([str(info) for info in exec_metadata[:-2]])
  with open(PATH_RESULTS + '%s.csv'%q_filename, 'w') as f:
    q.to_csv(f, index=False)


PATH_RESULTS = "results/"

# google_id = '1DgYTvfbzTe8CPMOwXMupmTOiMTP56PwD' #complete dataset
google_id = '1Jsg6ggQF98LT4D2QnEPTjj_XfTdeTWHh' #training dataset
gdd.download_file_from_google_drive(file_id=google_id, 
                                    dest_path = './data.csv', 
                                    showsize = True)
data = pd.read_csv("data.csv")

H = pd.DataFrame()
H['case'] = data['Case ID']
H['state'] = data['Activity'] + "-" + data['Workload']
H['action'] = data['Resource']
H['cost'] = data['Duration(s)']
H.head()

H_original = H.copy()

ALG = 'FQI'
GAMMA = 0.9
MAX_IT = 100
MAX_DIFF = 0
MAX_TIME = 10*60*60 #treinamento de no max 10h


h = H.copy()
MAX_DIFF = H.cost.quantile(.25)
q = init_q(h)
q['action'] = q['action'].astype(str) # para que o get_dummies funcione corretamente para essa coluna

x = (q['state'].apply(lambda x: pd.Series(literal_eval(x.split('-')[1])))
               .replace('FREE',0).replace('LOW',1).replace('HIGH',2))
x.columns = ['WL_'+str(col) for col in x.columns]
activities_dummies = pd.get_dummies(q['state'].apply(lambda x: x.split('-')[0]))
x = x.join(activities_dummies)
resources_dummies = pd.get_dummies(q['action'])
x = x.join(resources_dummies)

x.head()

h.cost.max()


"""#### FQI - LinearRegression"""

q, it, train_time = train(LinearRegression(), x, init_q(h))
save_exec(ALG, 'none', 'linear_regression_arrumado', 'default', GAMMA, it, train_time, q)

"""#### FQI - Random Forest"""

q, it, train_time = train(RandomForestRegressor(n_estimators=5), x, init_q(h))
save_exec(ALG, 'none', 'random_forest', 'n_estimators=5', GAMMA, it, train_time, q)

q, it, train_time = train(RandomForestRegressor(n_estimators=10), x, init_q(h))
save_exec(ALG, 'none', 'random_forest', 'n_estimators=10', GAMMA, it, train_time, q)

q, it, train_time = train(RandomForestRegressor(n_estimators=20), x, init_q(h))
save_exec(ALG, 'none', 'random_forest', 'n_estimators=20', GAMMA, it, train_time, q)

q, it, train_time = train(RandomForestRegressor(n_estimators=50), x, init_q(h))
save_exec(ALG, 'none', 'random_forest', 'n_estimators=20', GAMMA, it, train_time, q)

q, it, train_time = train(RandomForestRegressor(), x, init_q(h))
save_exec(ALG, 'none', 'random_forest', 'default', GAMMA, it, train_time, q)

"""#### NFQ - MLP"""

q, it, train_time = train(MLPRegressor(learning_rate='adaptive', alpha=0.005, hidden_layer_sizes=(10,10)), x, init_q(h))
save_exec(ALG, 'none', 'mlp', 'learning_rate=adaptive, alpha=0.005, hidden_layer_sizes=(10,10)', GAMMA, it, train_time, q)

q, it, train_time = train(MLPRegressor(learning_rate='adaptive', alpha=0.005, hidden_layer_sizes=(50,50)), x, init_q(h))
save_exec(ALG, 'none', 'mlp', 'learning_rate=adaptive, alpha=0.005, hidden_layer_sizes=(50,50)', GAMMA, it, train_time, q)

# Mesmo com max_iter=1000, nao converge
q, it, train_time = train(MLPRegressor(learning_rate='adaptive', max_iter = 1000), x, init_q(h))
save_exec(ALG, 'none', 'mlp', 'learning_rate=adaptive', GAMMA, it, train_time, q)

q, it, train_time = train(MLPRegressor(learning_rate = 'adaptive', activation = 'logistic'), x, init_q(h))
save_exec(ALG, 'none', 'mlp', 'learning_rate=adaptive, activation=logistic', GAMMA, it, train_time, q)

q, it, train_time = train(MLPRegressor(learning_rate = 'adaptive', activation = 'logistic', hidden_layer_sizes=(50,50)), x, init_q(h))
save_exec(ALG, 'none', 'mlp', 'learning_rate=adaptive, activation=logistic, hidden_layers=(50,50)', GAMMA, it, train_time, q)

"""#### NFQ - RProp"""

#Pra comparar com as primeiras execucoes de MLP, que usa Relu por padrao
network = Input(x.shape[1]) >> Relu(50) >> Relu(50) >> Relu(1)
q, it, train_time = train(algorithms.RPROP(network), x, init_q(h))
save_exec(ALG, 'none', 'RPROP', '50,50_relu', GAMMA, it, train_time, q)

#network = Input(x.shape[1]) >> Sigmoid(500) >> Sigmoid(500) >> Sigmoid(1)
q, it, train_time = train(algorithms.RPROP(network), x, init_q(h))
save_exec(ALG, 'none', 'RPROP', '500,500-Sigmoid', GAMMA, it, train_time, q)

#network = Input(x.shape[1]) >> Sigmoid(100) >> Sigmoid(1)
q, it, train_time = train(algorithms.RPROP(network), x, init_q(h))
save_exec(ALG, 'none', 'RPROP', '100-Sigmoid', GAMMA, it, train_time, q)