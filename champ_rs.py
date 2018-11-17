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

fit_model = False
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
# kNN recommendation system, given current team stat, return top 5 choices
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
            

# Synergy and Counter recommendation system, computing top 5 choices from Synergy and Counter scores
def s_and_c_rs(blue_team, red_team, team_side, blue_ban, red_ban, champ_row_name_dic):
    sy_dict = np.load('./fea_data/heroes_synergies.npy')
    ct_dict = np.load('./fea_data/heroes_counters.npy')

    selected_heroes_dict = {}

    for b in blue_team:
        selected_heroes_dict[champ_row_name_dic[b]] = True

    for r in red_team:
        selected_heroes_dict[champ_row_name_dic[r]] = True

    for b in blue_ban:
        selected_heroes_dict[champ_row_name_dic[b]] = True

    for r in red_ban:
        selected_heroes_dict[champ_row_name_dic[r]] = True

    if team_side == 'b':
        return top_5(selected_heroes_dict, sy_dict, ct_dict, blue_team, red_team, champ_row_name_dic)
    else:
        return top_5(selected_heroes_dict, sy_dict, ct_dict, red_team, blue_team, champ_row_name_dic)


def top_5(selected_heroes_dict, sy_dict, ct_dict, join_team, enemy_team, champ_row_name_dic):
    hero_num = 138
    overall_dict_list = []

    for i in range(hero_num):
        try:
            selected_heroes_dict[i]
        except:
            hero_sy_dict = sy_dict[i]
            hero_ct_dict = ct_dict[i]
            hero_ave_sy_towards_join_team = 0
            hero_ave_ct_towards_enemy_team = 0

            for hero in join_team:
                hero_ave_sy_towards_join_team += hero_sy_dict[hero]
            hero_ave_sy_towards_join_team /= len(join_team)

            for hero in enemy_team:
                hero_ave_ct_towards_enemy_team += hero_ct_dict[hero]
            hero_ave_ct_towards_enemy_team /= len(enemy_team)

            hero_overall = hero_ave_sy_towards_join_team + hero_ave_ct_towards_enemy_team

            hero_dict = {'overall_score': hero_overall, 'hero_index': i}

            overall_dict_list.append(hero_dict)

    overall_rank = sorted(overall_dict_list, key=lambda k: k['overall_score'])

    return [champ_row_name_dic[overall_rank[-1]['hero_index']],
            champ_row_name_dic[overall_rank[-2]['hero_index']],
            champ_row_name_dic[overall_rank[-3]['hero_index']],
            champ_row_name_dic[overall_rank[-4]['hero_index']],
            champ_row_name_dic[overall_rank[-5]['hero_index']]]


# Synergy and Counter recommendation system, computing the overall best hero for initial default choice
def s_and_c_overall_best(champ_row_name_dic):
    sy_dict = np.load('./fea_data/heroes_synergies.npy')
    ct_dict = np.load('./fea_data/heroes_counters.npy')

    hero_num = 138
    hero_ave_sy = []
    hero_ave_ct = []
    heroes_overall = []

    for i in range(len(sy_dict) if len(sy_dict) < len(ct_dict) else len(ct_dict)):
        for j in range(len(sy_dict[i]) if len(sy_dict[i]) < len(ct_dict[i]) else len(ct_dict[i])):
            hero_ave_sy[i] += sy_dict[i][j]
            hero_ave_ct[i] += ct_dict[i][j]
        hero_ave_sy[i] = hero_ave_sy[i]/(hero_num-1)
        hero_ave_ct[i] = hero_ave_ct[i]/(hero_num-1)
        hero_overall = hero_ave_sy[i] + hero_ave_ct[i]
        hero_dict = {'overall_score': hero_overall, 'hero_index': i}
        heroes_overall.append(hero_dict)

    overall_rank = sorted(heroes_overall, key=lambda k: k['overall_score'])

    return [champ_row_name_dic[overall_rank[-1]['hero_index']],
            champ_row_name_dic[overall_rank[-2]['hero_index']],
            champ_row_name_dic[overall_rank[-3]['hero_index']],
            champ_row_name_dic[overall_rank[-4]['hero_index']],
            champ_row_name_dic[overall_rank[-5]['hero_index']],
            champ_row_name_dic[overall_rank[-6]['hero_index']],
            champ_row_name_dic[overall_rank[-7]['hero_index']]]


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
    # dictionary: champ name -> id and id -> name
    champ_dic = {}
    # dictionary: champs.csv row num -> name and name -> champs.csv row num
    champ_row_name_dic = {}
    row_num = 0
    for index, row in champ_df.iterrows():
        champ_dic[row[0]] = row[1]
        champ_dic[row[1]] = row[0]
        champ_row_name_dic[row_num] = row[0]
        champ_row_name_dic[row[0]] = row_num
        row_num += 1
    
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

    if fit_model:
        # model choice
        kNN = KNeighborsClassifier(n_neighbors=20, weights = 'uniform')
        #svc = SVC(kernel='rbf')
        
        # final model decision and save to pickle
        #kNN.fit(team_sample, win_sample)
        kNN.fit(team, win)
        pickle.dump(kNN, open('saved_model/kNN_model.save', 'wb'))
    
        print('Model fit and saved!')
        
        return
    
    # reload model from pickle to save time
    kNN = pickle.load(open('saved_model/kNN_model.save', 'rb'))
    
    print('Model reloaded!')
    
    # validation
    if run_valid:
        print('Validation started!')
        
        print(np.average(cross_validate(kNN, team_sample, win_sample, cv=5, scoring='accuracy')['test_score']))
        
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
