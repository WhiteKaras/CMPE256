# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:49:13 2018

@author: Karas
"""

import random
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

# check if input is valid
def valid_input(sentence, champ_dic, champ_pool):
    while True:
        champ_name = input(sentence)
        if champ_name not in champ_dic:
            print("Wrong champion name, please try again.")
            continue
        if champ_dic[champ_name] not in champ_pool:
            print("This champion has been picked or banned, please try again.")
            continue
        return champ_name

# print team info
def print_info(blue_team, red_team, blue_ban, red_ban, champ_dic):
    blue_info = []
    for v in blue_team:
        blue_info.append(champ_dic[v])
    red_info = []
    for v in red_team:
        red_info.append(champ_dic[v])
    print("\nBlue Team: "+str(blue_info))
    print("Red Team: "+str(red_info))
    print("Blue Ban: "+str(blue_ban))
    print("Red Ban: "+str(red_ban)+'\n')
    
# make decision based on three different rs algorithm
def champ_decision(kNN_result):
    return kNN_result[0]

##############
#RS Algorithm#
##############
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
    blue_ban = []
    red_ban = []

    while True:
        team_side = input('Please enter your side (b/r): ')
        if team_side != 'b' and team_side != 'r':
            print('This is not a valid side, please try again.')
            continue
        break
    
    # player as blue team 
    if team_side == 'b':
        # ban round
        # b3 ban
        ban = valid_input("Player B3 ban: ", champ_dic, champ_pool)
        blue_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        
        # r3 ban: random
        ban = champ_dic[random.choice(champ_pool)]
        print("Player R3 ban: "+ban)
        red_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        
        # b4 ban
        ban = valid_input("Player B4 ban: ", champ_dic, champ_pool)
        blue_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        
        # r4 ban: random
        ban = champ_dic[random.choice(champ_pool)]
        print("Player R4 ban: "+ban)
        red_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        
        # b5 ban
        ban = valid_input("Player B5 ban: ", champ_dic, champ_pool)
        blue_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        
        # r5 ban: random
        ban = champ_dic[random.choice(champ_pool)]
        print("Player R5 ban: "+ban)
        red_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        
        # pick round
        # b1 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # highest win rate champ for the first pick
        pick = 'Jax'
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B1' recommended pick: "+pick)
        input("Press enter to continue...")
        
        # r1 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R1 pick: "+pick)
        input("Press enter to continue...")
        
        # r2 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R2 pick: "+pick)
        input("Press enter to continue...")
        
        # b2 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        pick = champ_decision(kNN_result)
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B2' recommended pick: "+pick)
        input("Press enter to continue...")
        
        # b3 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        pick = champ_decision(kNN_result)
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B3' recommended pick: "+pick)
        input("Press enter to continue...")
        
        # r3 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R3 pick: "+pick)
        input("Press enter to continue...")
        
        # r4 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R4 pick: "+pick)
        input("Press enter to continue...")
        
        # b4 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        pick = champ_decision(kNN_result)
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B4' recommended pick: "+pick)
        input("Press enter to continue...")
        
        # b5 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        pick = champ_decision(kNN_result)
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B5' recommended pick: "+pick)
        input("Press enter to continue...")
        
        # r5 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R5 pick: "+pick)
        input("Press enter to continue...")
    
    # player as red team    
    else:
        # ban round
        # b3 ban: random
        ban = champ_dic[random.choice(champ_pool)]
        print("Player B3 ban: "+ban)
        blue_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        
        # r3 ban
        ban = valid_input("Player R3 ban: ", champ_dic, champ_pool)
        red_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        
        # b4 ban: random
        ban = champ_dic[random.choice(champ_pool)]
        print("Player B4 ban: "+ban)
        blue_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        
        # r4 ban
        ban = valid_input("Player R4 ban: ", champ_dic, champ_pool)
        red_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        
        # b5 ban: random
        ban = champ_dic[random.choice(champ_pool)]
        print("Player B5 ban: "+ban)
        blue_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        
        # r5 ban
        ban = valid_input("Player R5 ban: ", champ_dic, champ_pool)
        red_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        
        # pick round
        # b1 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B1 pick: "+pick)
        input("Press enter to continue...")
        
        # r1 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        pick = champ_decision(kNN_result)
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R1' recommended pick: "+pick)
        input("Press enter to continue...")
        
        # r2 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        pick = champ_decision(kNN_result)
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R2' recommended pick: "+pick)
        input("Press enter to continue...")
        
        # b2 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B2 pick: "+pick)
        input("Press enter to continue...")
        
        # b3 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B3 pick: "+pick)
        input("Press enter to continue...")
        
        # r3 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        pick = champ_decision(kNN_result)
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R3' recommended pick: "+pick)
        input("Press enter to continue...")
        
        # r4 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        pick = champ_decision(kNN_result)
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R4' recommended pick: "+pick)
        input("Press enter to continue...")
        
        # b4 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B4 pick: "+pick)
        input("Press enter to continue...")
        
        # b5 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B5 pick: "+pick)
        input("Press enter to continue...")
        
        # r5 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        pick = champ_decision(kNN_result)
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R5' recommended pick: "+pick)
        input("Press enter to continue...")
        

    print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)   
    print('Recommandation ended!')
    
main()
