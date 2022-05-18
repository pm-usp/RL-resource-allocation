import os
import sys
import json
import math
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
from google_drive_downloader import GoogleDriveDownloader as gdd
from ast import literal_eval

## FUNCTIONS

def get_duration_mean():
  google_id = '1TbJb2OBWLubEgWT-WEmn5s4hTwUSouw9'
  gdd.download_file_from_google_drive(file_id=google_id, 
                                      dest_path = './duration_predictor_mean.csv', 
                                      showsize = True)
  return pd.read_csv('duration_predictor_mean.csv')

def get_test_df():
  csv_filename = load_test_df()
  df = pd.read_csv(csv_filename)
  df = rename_cols(df)
  df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format="%Y/%m/%d %H:%M:%S")
  df = get_next_events(df)
  return df

def load_test_df():
  csv_filename = 'data_test.csv'
  google_id = '1881WtafRdH_hk2gUxH18a2xr9MkG6mJs'
  gdd.download_file_from_google_drive(file_id=google_id, 
                                      dest_path = './%s'%csv_filename, 
                                      showsize = True)
  return csv_filename

def rename_cols(df):
  rename_dict = {'lifecycle:transition_shifted': 'lifecycle:transition_prev',
                  'Activity_shifted':'Activity_prev',
                  'org:resource_shifted': 'org:resource_prev',
                  'time:timestamp_shifted': 'time:timestamp_prev'}
  return df.rename(columns=rename_dict)

def get_next_events(df):
  df = df.sort_values(['Case ID','Activity','org:resource','time:timestamp'])
  df['lifecycle:transition_next'] =  df['lifecycle:transition'].shift(-1)
  df['Activity_next'] = df['Activity'].shift(-1)
  df['org:resource_next'] = df['org:resource'].shift(-1)
  df['time:timestamp_next'] = df['time:timestamp'].shift(-1)
  df['time:timestamp_next'] = pd.to_datetime(df['time:timestamp_next'], format="%Y/%m/%d %H:%M:%S")
  df['Duration(s)_next'] = df['Duration(s)'].shift(-1)
  return df

def process_q(q, alg_name):

  def get_data_pi(n):

    def get_pi():
      pi_filepath = PATH_RESULTS + 'pis/'
      pi_filename = '%s-n_%s.csv'%(alg_name,str(n))
      if os.path.isfile(pi_filepath+pi_filename):
        return pd.read_csv(open(pi_filepath+pi_filename, 'r'))
      pi = get_pi(q, n)
      save_pi()
      return pi

    def save_pi():
      with open(PATH_RESULTS + 'pis/%s'%filename, 'w') as f:
        pi.to_csv(f, index=False)

    def save_data_pi():
      with open(PATH_RESULTS + 'pi_applied_data/%s'%filename, 'w') as f:
        data_pi[['Case ID', 'Activity', 'Duration(s)_next','pi_action','pred_duration','pi_action_dur_source', 'Workload', 'time:timestamp']].to_csv(f, index=False)
    
    filepath = PATH_RESULTS + 'pi_applied_data/'
    filename = '%s-n_%s.csv'%(alg_name,str(n))
    if os.path.isfile(filepath+filename):
      return pd.read_csv(open(filepath+filename, 'r'))
    
    pi = get_pi()
    data_pi = apply_pi(pi)
    save_data_pi()
  
    return data_pi

  ## MAIN OF PROCESS_Q FUNCTION

  global QS, DELTAS, STATS, CASES_OTIMIZ

  d_melted_n = q.melt(id_vars=['case','state','action','cost', 'next_state'])
  d_melted_n = d_melted_n.rename(columns={'variable':'N', 'value':'Q'})
  d_melted_n['N'] = d_melted_n['N'].astype(int)
  d_melted_n['Q'] = d_melted_n['Q']/3600 #em horas
  d_melted_n['Modelo'] = alg_name
  QS = pd.concat([QS, d_melted_n])

  diffs = {'N':[], 'Delta':[]}
  q_stats = {'Total(d)': [], 'Otimizados (%)': [], 'N':[], 'Modelo': [], 
             '#Original_dur':[], '#Dur_model':[], '#Dur_avg_wl':[], '#Dur_avg_act':[], '#Events Otimiz':[]}
  data_test_pi = None
  last_n = d_melted_n.N.max()
  for n in range(1, last_n+1):
    diffs['N'].append(n)
    diffs['Delta'].append(abs(q[str(n)]-q[str(n-1)]).sum()/3600)

    data_test_pi = get_data_pi(n)

    cases_otimiz = pd.DataFrame(data_test_pi.groupby('Case ID')['Duration(s)_next'].sum().reset_index())
    cases_otimiz['Pi'] = data_test_pi.groupby('Case ID')['pred_duration'].sum().reset_index()['pred_duration']
    cases_otimiz['Otimizacao'] = cases_otimiz['Pi']-cases_otimiz['Duration(s)_next']

    q_stats['Total(d)'].append(data_test_pi.pred_duration.sum()/3600/24),
    q_stats['Otimizados (%)'].append((cases_otimiz['Otimizacao'] > 0).sum()/len(cases_otimiz['Case ID'].unique()))
    q_stats['#Original_dur'].append(data_test_pi[data_test_pi['pi_action_dur_source'] == 'original'].shape[0])
    q_stats['#Dur_model'].append(data_test_pi[data_test_pi['pi_action_dur_source'] == 'ml_model'].shape[0])
    q_stats['#Dur_avg_wl'].append(data_test_pi[data_test_pi['pi_action_dur_source']=='avg_wl'].shape[0])
    q_stats['#Dur_avg_act'].append(data_test_pi[data_test_pi['pi_action_dur_source']=='avg_act'].shape[0])
    data_test_pi['pi-original_duration'] = data_test_pi['pred_duration'] - data_test_pi['Duration(s)_next']
    q_stats['#Events Otimiz'].append(data_test_pi[data_test_pi['pi-original_duration'] > 0].shape[0]/data_test_pi.shape[0])
    q_stats['N'].append(n)
    q_stats['Modelo'].append(alg_name)

  STATS = pd.concat([STATS, pd.DataFrame(q_stats)])
  
  diffs = pd.DataFrame(diffs)
  diffs['Modelo'] = alg_name
  DELTAS = pd.concat([DELTAS, diffs])
  
  cases_otimiz['Modelo'] = alg_name
  cases_otimiz.rename(columns={'Duration(s)_next':'Custo original'})
  CASES_OTIMIZ = pd.concat([CASES_OTIMIZ, cases_otimiz])

def get_pi(states_set, n):
  aux = pd.DataFrame(states_set.groupby(['state','action']).apply(lambda x: x[str(n)].mean())).reset_index()
  return aux.loc[aux.groupby(['state'])[0].idxmin()]

def apply_pi(pi):
  def inicialize_workload_count():
    WORKLOAD_COUNT = pd.DataFrame(columns=RESOURCES)
    WORKLOAD_COUNT.loc[0] = 0
    #partir do Workload_count final do treinamento
    wc_final_train = LAST_TRAINING_WL
    for r,count in wc_final_train.items():
      WORKLOAD_COUNT[r].loc[0] = count
    return WORKLOAD_COUNT

  def process_event(event):

    def set_event_wl():
      wl = get_wl(WORKLOAD_COUNT)
      event['Workload'] = get_wl_dict(wl)
      return wl


    def get_wl(wl_count):
      activities_being_exec = wl_count.T.sum().loc[0]
      AVG_R = (activities_being_exec/len(wl_count.columns)) 

      def get_scale_int(x):
        if x[0] < 1:
          x[0] = 0#'FREE'
          return x
        if x[0] <= AVG_R:
          x[0] = 1#'LOW'
          return x
        x[0] = 2#'HIGH'
        return x
      
      wl = wl_count.copy().T
      wl = wl.apply(get_scale_int, axis=1)
      return wl.T


    def get_wl_dict(wl):
      return wl.to_dict('records')[0]


    def get_original_duration():
      original_duration = None
      if (event['lifecycle:transition_next'] == 'COMPLETE' and 
          event['Activity_next'] == event['Activity'] and
          event['org:resource_next'] == event['org:resource']):
        # verified that the dilemma of having two activity instances started for the same case (and executed by the same resource), 
        #  described in Process Mining book (by W. van der Aalst), p.132, does not happen in this log
        original_duration = event['Duration(s)_next'] 
      return original_duration


    def set_action_and_duration(wl):
      s = event['Activity'] + '-' + get_wl_str(event['Workload'])
      a_pi, duration, source = get_pi_action_and_duration(s, event['org:resource'], original_duration, 
                                                  event['Activity'], wl)
      event['pi_action'] = a_pi
      event['pred_duration'] = duration
      event['pi_action_dur_source'] = source


    def get_wl_str(wl):
      return json.dumps(wl).replace('"','').replace(': 0',": 'FREE'").replace(': 1',": 'LOW'").replace(': 2',": 'HIGH'")


    def get_pi_action_and_duration(state, original_action, original_duration, activity, wl):
      selection = pi[pi['state'] == state].action
      if selection.shape[0]>0:
        a_pi = int(selection.iloc[0])
        WORKLOAD_COUNT[a_pi] = WORKLOAD_COUNT[a_pi] + 1
        duration, source = predict_duration(a_pi, activity, wl)
        return a_pi, duration, source
      WORKLOAD_COUNT[int(original_action)] = WORKLOAD_COUNT[int(original_action)] + 1
      return original_action, original_duration, 'original'


    def predict_duration(action, activity, wl):
      x_predictor = base.copy() #indicar a atividade e o recurso 
      x_predictor.loc[float(action)] = 1
      x_predictor.loc[activity] = 1
      x_predictor.loc['workload'] = wl[action].iloc[0]
      select_mean = duration_predictor_mean[(duration_predictor_mean['org:resource']==action) &
                            (duration_predictor_mean['Activity']==activity) &
                            (duration_predictor_mean['Workload-esp_resource']==wl[action].iloc[0])]
      if select_mean.shape[0]>0:
        return select_mean['mean'].iloc[0], 'avg_wl'
      return duration_predictor_mean[(duration_predictor_mean['org:resource']==action) &
                              (duration_predictor_mean['Activity']==activity)].mean.mean(), 'avg_act'


    def create_sint_complete_event():
      global DATA2PROCESS
      end_event = event.copy()
      event_complete_ts = event['time:timestamp'] + pd.to_timedelta(event['pred_duration'], unit = 'seconds')
      end_event['time:timestamp'] = event_complete_ts
      end_event['lifecycle:transition'] = 'SINT_COMPLETE' 
      DATA2PROCESS = DATA2PROCESS.append(end_event)

      return event_complete_ts


    def update_case_events_ts(event_complete_ts, original_duration):
      case_mask = (DATA2PROCESS['Case ID'] == event['Case ID'])
      ts_mask = (DATA2PROCESS['time:timestamp'] > event_complete_ts)

      durat_diff = 0

      durat_diff = original_duration - event['pred_duration']
      if durat_diff != 0:
        DATA2PROCESS.loc[case_mask & ts_mask, 'time:timestamp'] = DATA2PROCESS.loc[case_mask & ts_mask, 'time:timestamp'] - pd.to_timedelta(durat_diff, unit = 'seconds')

    # MAIN OF PROCESS_EVENT METHOD
    if event['lifecycle:transition'] == 'START':
      wl = set_event_wl()
      original_duration = get_original_duration()
      set_action_and_duration(wl) #from pi
      event_complete_ts = create_sint_complete_event()
      update_case_events_ts(event_complete_ts, original_duration)

    if event['lifecycle:transition'] == 'SINT_COMPLETE':
      WORKLOAD_COUNT[int(event['pi_action'])] = WORKLOAD_COUNT[int(event['pi_action'])] - 1
    
    return event

  #MAIN OF APPLY_PI METHOD
  global DATA2PROCESS
  WORKLOAD_COUNT = inicialize_workload_count()
  DATA2PROCESS = data_test.sort_values('time:timestamp').copy()
  PROCESSED = pd.DataFrame()

  while DATA2PROCESS.shape[0]>1:
    PROCESSED = PROCESSED.append(process_event(DATA2PROCESS.iloc[0]))
    DATA2PROCESS = DATA2PROCESS.iloc[1:].sort_values('time:timestamp')

  return PROCESSED[PROCESSED['lifecycle:transition']!= 'SINT_COMPLETE']


## MAIN

PATH_RESULTS = 'results/'

warnings.filterwarnings('ignore')

duration_predictor_mean = get_duration_mean()
data_test = get_test_df()

LAST_TRAINING_WL = {10138: 0, 10609: 0, 10629: 0, 10809: 0, 10861: 0, 10881: 0, 10889: 0, 
                      10899: 0, 10909: 0, 10910: 1, 10913: 0, 10929: 0, 10932: 0, 10972: 0, 
                      10982: 0, 11000: 1, 11003: 1, 11009: 0, 11049: 0, 11119: 0, 11121: 0, 
                      11122: 0, 11169: 0, 11179: 0, 11180: 0, 11181: 0, 11189: 0, 11201: 0, 
                      11203: 0, 11259: 0}

RESOURCES = [11180,10982,11121,10609,10899,10629,11049,11201,10889,11119,11179,11169,10809,11122,
             11181,11009,11189,10138,10881,10909,10972,11203,10913,11000,10861,11259,10932,10910,
             10929,11003]

data_test = data_test[((data_test['lifecycle:transition'] == 'START') & (~data_test['Duration(s)_next'].isnull())) | 
                      (data_test['lifecycle:transition'] == 'COMPLETE')]

results = [('1M0wD8lsRAKUwFBufJfeid77TYi-pnY2A', 'RL'),
           ('1-1ipoKUqD6lP1qaD-JK-aiInwhNy_kiC', 'RF - 5'),
           ('1-4ZSsOBsDpE35bmFEGMZkc-LCFk50sJ-', 'RF - 10'),
           ('1-4yFL-z8EhVCWSUSmBpJp3YvhhxJco3Y', 'RF - 20'),
           ('1-98yguJIuLUpAEf412DizIwCm0KM2nZt', 'RF - 100'),
           ('1-EynA-s8So8CW8o4d8ojk00Z40y8MRZP', 'MLP - 10,10_relu'),
           ('1-HIUVV-XIpBZ_kU4xhOhFWu4-GDmkFs5', 'MLP - 50,50_relu'),
           ('1-IZUp5jSg9Pq3KrcGSU8WI5HBGkdhOP1', 'MLP - 100_relu'),
           ('1-_RrBUdxCaCnpj_yu-9ux8C6OEAq-SGD', 'MLP - 50,50_sigmoide'),
           ('1-YYkvxn-HvqSM1QGePWcn2cxhvKfBOr1', 'MLP - 100_sigmoide'),
           ('1-beS5nmVNjvPzXj7v6b_t5PfL76UVhTC', 'RPROP - 50,50_relu'),
           ('1-eSmFcFTl2VjORoMVrhd_613YnVdyJxx', 'RPROP - 100_sigmoide'),
           ('1-dL6U7hYb3XVzjmF1EEtQZlNUsrqR8sL', 'RPROP - 500,500_sigmoide'),]

QS = pd.DataFrame()
STATS = pd.DataFrame()
DELTAS = pd.DataFrame()
CASES_OTIMIZ = pd.DataFrame()

for q_g_id, alg in results:
    print(alg)
    gdd.download_file_from_google_drive(file_id=q_g_id, 
                                    dest_path = './%s.csv'%alg, 
                                    showsize = True)
    q = pd.read_csv('./%s.csv'%alg)
    process_q(q, alg)

QS.to_csv('QS.csv')
STATS.to_csv('STATS.csv')
DELTAS.to_csv('DELTAS.csv')
CASES_OTIMIZ.to_csv('CASES_OTIMIZ.csv')