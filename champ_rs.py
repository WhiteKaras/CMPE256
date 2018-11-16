# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:49:13 2018

@author: Karas
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
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
def champ_one_hot(team_list,champ_id,team, old_result):
    result = old_result.copy()
    if len(result)==0:
        for v in champ_id:
            result.append(0)
            result.append(0)
    for v in team_list:
        for i in range(len(champ_id)):
            if champ_id[i]==v:
                if team == 'b':
                    result[i*2]=1
                else:
                    result[i*2+1]=1
                break
    return result

# kNN recommandation system, given current team stat, return top 5 choices
def kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic):
    # fixed team
    current_team = champ_one_hot(blue_team, champ_id, 'b', [])
    current_team = champ_one_hot(red_team, champ_id, 'r', current_team)
    
    # pick different champ and rank their winning rate
    rank = []
    for champ in champ_pool:
        # pick for blue team
        if team_side == 'b':
            temp_team = champ_one_hot([champ], champ_id, 'b', current_team)
        # pick for red team
        else:
            temp_team = champ_one_hot([champ], champ_id, 'r', current_team)
        
        # get score for this champ
        k_list = kNN.kneighbors([temp_team], return_distance=False)[0]
        score = 0
        if team_side == 'b':
            for i in k_list:
                score += win[i]
        else:
            for i in k_list:
                score += win[i]^1
        score /= len(k_list)
        rank.append((champ, score))
        
    # return top 5 score
    rank.sort(key=lambda v:v[1])
    return [champ_dic[rank[0][0]],
            champ_dic[rank[1][0]],
            champ_dic[rank[2][0]],
            champ_dic[rank[3][0]],
            champ_dic[rank[4][0]]]    
            

######                
#main#    
######
def main():
    # read champ_df
    champ_df = pd.read_csv(
        filepath_or_buffer='ori_data/champs.csv', 
        header=0,
        sep=',')
    # champ_id list
    champ_id = champ_df.id.tolist()
    #  dictionary: champ name -> id and id -> name
    champ_dic = {}
    for index, row in champ_df.iterrows():
        champ_dic[row[0]] = row[1]
        champ_dic[row[1]] = row[0]
    
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
    
    # validation sample
    team_sample = team[0:10000]
    win_sample = win[0:10000]

    # model choice
    kNN = KNeighborsClassifier(n_neighbors=10, weights = 'uniform')
    #svc = SVC(kernel='rbf')
    
    # final model decision
    kNN.fit(team_sample, win_sample)
    
    print('Model fit!')
    
    # validation
    if run_valid:
        print('Validation started!')
        
        print(np.average(cross_validate(kNN, team_sample, win_sample, cv=4, scoring='accuracy')['test_score']))
        
        print('Validation finished!')
        return
    
    # BP recommendation    
    print('Recommendation started!')
    
    champ_pool = champ_id.copy()
    blue_team = []
    red_team = []
    
    champ_pool.remove(champ_dic['Vayne'])
    
    blue_team.append(champ_dic['Jax'])
    blue_team.append(champ_dic['Sona'])
    blue_team.append(champ_dic['Tristana'])
    red_team.append(champ_dic['Varus'])
    red_team.append(champ_dic['Fiora'])
    red_team.append(champ_dic['Singed'])
    
    team_side = 'b'
    
    if team_side == 'b':
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
    
        
    print('Recommandation ended!')
    
main()
