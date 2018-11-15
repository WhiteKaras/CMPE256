# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:49:13 2018

@author: Karas
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import pickle

################
#Global Setting#
################

run_valid = False

##########
#function#
##########
# one hot encode for champ_id
def champ_one_hot(l1,l2,team, result):
    if len(result)==0:
        for v in l2:
            result.append(0)
            result.append(0)
    for v in l1:
        for i in range(len(l2)):
            if l2[i]==v:
                if team == 'b':
                    result[i*2]=1
                else:
                    result[i*2+1]=1
                break
    return result

######                
#main#    
######
def main():
    # read champ_df
    champ_df = pd.read_csv(
        filepath_or_buffer='ori_data/champs.csv', 
        header=0,
        sep=',')
    
    #  dictionary: champ name -> id
    champ_id = {}
    for index, row in champ_df.iterrows():
        champ_id[row[0]] = row[1]
    
    # read s8_champ_win_data.csv as nparray
    df = pd.read_csv(
        filepath_or_buffer='pre_data/s8_champ_win_data.csv', 
        header=0,
        sep=',')
    df = df.values
    
    # seperate team and win
    team = []
    win = []
    for v in df:
        team.append(v[0:276])
        win.append(v[276])
    
    print('Data prepared!')
    
    team_sample = team[0:10000]
    win_sample = win[0:10000]

    # model choice
    kNN = KNeighborsClassifier(n_neighbors=10, weights = 'distance')
    svc = SVC(kernel='rbf')
    
    # final model decision
    model = svc.fit(team, win)
    
    print('Model fit!')
    
    # validation
    if run_valid:
        print('Validation started!')
        
        print(np.average(cross_validate(model, team_sample, win_sample, cv=4, scoring='accuracy')['test_score']))
        
        print('Validation finished!')
    
    # recommandation    
    else:
        print('Recommandation started!')
        #model.fit(team, win)
    
main()
