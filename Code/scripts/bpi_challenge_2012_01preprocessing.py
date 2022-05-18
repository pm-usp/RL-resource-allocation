import numpy as np
import pandas as pd
import pickle
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge 
from sklearn.neural_network import MLPRegressor 
from sklearn.metrics import r2_score
from google_drive_downloader import GoogleDriveDownloader as gdd


google_id = "1wWxY2N737gjof4FQkTnaBAQaQu8_4g6L"
gdd.download_file_from_google_drive(file_id=google_id, 
                                    dest_path = './data.csv', 
                                    showsize = True)
log = pd.read_csv("data.csv")

data = log.copy()

data['Activity'] = data['Activity'].replace({
                    'W_Nabellen incomplete dossiers': 'W_Calling to add missing information to the application',
                    'W_Afhandelen leads':'W_Fixing incoming lead',
                    'W_Completeren aanvraag':'W_Filling in information for the application',
                    'W_Nabellen offertes': 'W_Calling after sent offers',
                    'W_Valideren aanvraag': 'W_Assessing the application',
                    'W_Beoordelen fraude': 'W_Evaluate fraud',
                    'W_Wijzigen contractgegevens': 'W_Change contract details'})

data = data[data['Activity'].apply(lambda x: 'W_' in x)]

data[data['lifecycle:transition'] == 'START'].to_csv('data_logformat.csv', index = False)

missing_resource_cases = data[data["org:resource"].isnull()]['Case ID'].unique()
data = data[~data['Case ID'].isin(missing_resource_cases)]
data = data[data['org:resource'] != 112]
data = data[data['lifecycle:transition']!='SCHEDULE']

data.to_csv('bpi12_filtered.csv',index=False)

act_count = pd.DataFrame(data.groupby('org:resource')['Activity'].count().sort_values()).reset_index()
resources2del = act_count[act_count.Activity < act_count.Activity.mean()]['org:resource']
cases2del = data[data['org:resource'].isin(resources2del)]['Case ID'].unique()
data = data[~data['Case ID'].isin(cases2del)]

data['time:timestamp'] = pd.to_datetime(data['time:timestamp'], format="%d/%m/%Y %H:%M:%S")

data = data.sort_values(['Case ID','Activity','org:resource','time:timestamp'])

def calc_duration(x):
  if (x['lifecycle:transition'] == 'COMPLETE' and 
      x['lifecycle:transition_shifted'] == 'START' and 
      x['Activity_shifted'] == x['Activity'] and
      x['org:resource_shifted'] == x['org:resource']):
    return x['time:timestamp'] - x['time:timestamp_shifted']
  return np.nan

data['lifecycle:transition_shifted'] = data['lifecycle:transition'].shift()
data['Activity_shifted'] = data['Activity'].shift()
data['org:resource_shifted'] = data['org:resource'].shift()
data['time:timestamp_shifted'] = data['time:timestamp'].shift()
data['duration'] = data.apply(calc_duration, axis=1)
data['Duration(s)'] = data.duration.apply(lambda x: pd.Timedelta(x).total_seconds())

## DIVIDE TRAIN AND TEST SETS

division_point = data['time:timestamp'].quantile(0.75)
data_train = data[data['time:timestamp'] < division_point]
data_test = data[data['time:timestamp'] >= division_point]

data_train['Workload_count'] = [{int(resource): 0 for resource in data_train['org:resource'].unique()}]*len(data_train)

RESOURCES_WORKLOAD = {int(resource): 0 for resource in data_train['org:resource'].unique()}
def update_workload_count(x):
  if x['lifecycle:transition'] == 'START': 
    RESOURCES_WORKLOAD[int(x['org:resource'])] += 1
  elif x['lifecycle:transition'] == 'COMPLETE' and RESOURCES_WORKLOAD[int(x['org:resource'])] > 0:
    RESOURCES_WORKLOAD[int(x['org:resource'])] -= 1
  x.Workload_count = RESOURCES_WORKLOAD.copy()
  return x

data_train = data_train.sort_values(by='time:timestamp').apply(lambda x: update_workload_count(x), axis=1)

def def_workload_status(x):
  activities_being_exec = np.array(list(x['Workload_count'].values())).sum()
  AVG_R = (activities_being_exec/qtd_resources) 
  def get_r_workload(resource):
    if x['Workload_count'][resource] < 1:
      return 'FREE'
    elif x['Workload_count'][resource] <= AVG_R: 
      return 'LOW'
    return 'HIGH' 
  x['Workload'] = {resource: get_r_workload(resource) for resource in x['Workload_count']}
  return x
data_train = data_train.apply(lambda x: def_workload_status(x), axis=1)

data_train = data_train.dropna()

data_train_final = data_train[['Case ID', 'Activity', 'org:resource', 'Workload', 'Duration(s)']]
data_train_final['org:resource'] = data_train_final['org:resource'].astype(int).astype(str)
data_train_final = data_train_final.rename(columns = {'org:resource':'Resource'})

data_train_final.to_csv('bpi12_preprocessed_train.csv',index=False)
data_test.to_csv('bpi12_preprocessed_test.csv',index=False)