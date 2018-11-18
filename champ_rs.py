# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:49:13 2018

@author: Karas
"""

import random
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
#from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import pickle

################
#Global Setting#
################

fit_kNN_model = False
fit_NB_model = False
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
    print("Blue Team: "+str(blue_info))
    print("Red Team: "+str(red_info))
    print("Blue Ban: "+str(blue_ban))
    print("Red Ban: "+str(red_ban)+'\n')
    
# make decision based on three different rs algorithm
def champ_decision(kNN_result, sc_result, NB_result):
    d = {}
    # weigh to each rs list
    weigh(d, kNN_result)
    weigh(d, sc_result)
    weigh(d, NB_result)
        
    return max(d, key=d.get)

# weigh logic:  first place less than 2, last place 1.0
def weigh(d, key):
    for i in range(len(key)):
        if key[i] not in d:
            d[key[i]] = 2-1/len(key)*i
        else:
            d[key[i]] += 2-1/len(key)*i

##############
#RS Algorithm#
##############
# kNN recommendation system, given current team stat, return top choices
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
        
    # return top score
    rank.sort(key=lambda v:v[1], reverse=True)

    return [champ_dic[rank[i][0]] for i in range(20)]
            

# Synergy and Counter recommendation system, computing top choices from Synergy and Counter scores
def s_and_c_rs(blue_team, red_team, team_side, blue_ban, red_ban, champ_dic, champ_row_name_dic):
    sy_dict = np.load('./fea_data/heroes_synergies.npy')
    ct_dict = np.load('./fea_data/heroes_counters.npy')
    b_team = []
    r_team = []

    selected_heroes_dict = {}

    for b in blue_team:
        b_team.append(champ_row_name_dic[champ_dic[b]])
        selected_heroes_dict[champ_row_name_dic[champ_dic[b]]] = True

    for r in red_team:
        r_team.append(champ_row_name_dic[champ_dic[r]])
        selected_heroes_dict[champ_row_name_dic[champ_dic[r]]] = True

    for b in blue_ban:
        selected_heroes_dict[champ_row_name_dic[b]] = True

    for r in red_ban:
        selected_heroes_dict[champ_row_name_dic[r]] = True

    if team_side == 'b':
        return top_5(selected_heroes_dict, sy_dict, ct_dict, b_team, r_team, champ_row_name_dic)
    else:
        return top_5(selected_heroes_dict, sy_dict, ct_dict, r_team, b_team, champ_row_name_dic)


def top_5(selected_heroes_dict, sy_dict, ct_dict, join_team, enemy_team, champ_row_name_dic):
    hero_num = 138
    overall_dict_list = []

    for i in range(hero_num):
        try:
            selected_heroes_dict[i]
        except:
            hero_sy_dict = sy_dict[i]
            hero_ct_dict = ct_dict[i]
            hero_ave_sy_towards_join_team = 1
            hero_ave_ct_towards_enemy_team = 1

            for hero in join_team:
                hero_ave_sy_towards_join_team += hero_sy_dict[hero] + 2
            if len(join_team)!= 0:
                hero_ave_sy_towards_join_team /= len(join_team)

            for hero in enemy_team:
                hero_ave_ct_towards_enemy_team += hero_ct_dict[hero]
            if len(enemy_team)!= 0:
                hero_ave_ct_towards_enemy_team /= len(enemy_team)

            hero_overall = hero_ave_sy_towards_join_team * hero_ave_ct_towards_enemy_team

            hero_dict = {'overall_score': hero_overall, 'hero_index': i}

            overall_dict_list.append(hero_dict)

    overall_rank = sorted(overall_dict_list, key=lambda k: k['overall_score'], reverse=True)

    return [champ_row_name_dic[overall_rank[i]['hero_index']] for i in range(20)]


# Synergy and Counter recommendation system, computing the overall best hero for initial default choice
def s_and_c_overall_best(champ_row_name_dic):
    self_wrate = np.load('./fea_data/heroes_self_win_rate.npy')

    hero_num = 138
    heroes_overall = []

    for i in range(hero_num):
        hero_dict = {'self_win_rate': self_wrate[i], 'hero_index': i}
        heroes_overall.append(hero_dict)

    overall_rank = sorted(heroes_overall, key=lambda k: k['self_win_rate'])

    return [champ_row_name_dic[overall_rank[-1]['hero_index']],
            champ_row_name_dic[overall_rank[-2]['hero_index']],
            champ_row_name_dic[overall_rank[-3]['hero_index']],
            champ_row_name_dic[overall_rank[-4]['hero_index']],
            champ_row_name_dic[overall_rank[-5]['hero_index']],
            champ_row_name_dic[overall_rank[-6]['hero_index']],
            champ_row_name_dic[overall_rank[-7]['hero_index']]]


# NB recommendation algorithm
def NB_rs(blue_team, red_team, team_side, champ_id, champ_pool, NB, win, champ_dic):
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
        if team_side == 'b':
            rank.append((champ, NB.predict_log_proba([temp_team])[0][1]-NB.predict_log_proba([temp_team])[0][0]))
        else:
            rank.append((champ, NB.predict_log_proba([temp_team])[0][0]-NB.predict_log_proba([temp_team])[0][1]))
        
    # return top score
    rank.sort(key=lambda v:v[1], reverse=True)
    return [champ_dic[rank[i][0]] for i in range(20)]
            

# Winning rate predictor after both teams have been fulfilled
def win_rate_predictor(blue_team, red_team, team_side, champ_dic, champ_row_name_dic):
    b_team = []
    r_team = []
    sy_dict = np.load('./fea_data/heroes_synergies.npy')
    ct_dict = np.load('./fea_data/heroes_counters.npy')
    self_wrate = np.load('./fea_data/heroes_self_win_rate.npy')

    for b in blue_team:
        b_team.append(champ_row_name_dic[champ_dic[b]])

    for r in red_team:
        r_team.append(champ_row_name_dic[champ_dic[r]])


# SC predict
def sv_predict(blue_team, red_team, champ_row_name_dic):
    sy_dict = np.load('./fea_data/heroes_synergies.npy')
    ct_dict = np.load('./fea_data/heroes_counters.npy')
    
    blue_team_score = 0
    red_team_score = 0
    
    # calculate blue team score
    for current_hero in blue_team:
        hero_ave_sy_towards_join_team = 0
        hero_ave_ct_towards_enemy_team = 0
        
        for hero in blue_team:
            hero_ave_sy_towards_join_team += sy_dict[current_hero][hero] + 2
            hero_ave_sy_towards_join_team /= 5
    
        for hero in red_team:
            hero_ave_ct_towards_enemy_team += ct_dict[current_hero][hero]
            hero_ave_ct_towards_enemy_team /= 5
    
        blue_team_score += hero_ave_sy_towards_join_team * hero_ave_ct_towards_enemy_team
        
    # calculate red team score
    for current_hero in red_team:
        hero_ave_sy_towards_join_team = 0
        hero_ave_ct_towards_enemy_team = 0
        
        for hero in red_team:
            hero_ave_sy_towards_join_team += sy_dict[current_hero][hero] + 2
            hero_ave_sy_towards_join_team /= 5
    
        for hero in blue_team:
            hero_ave_ct_towards_enemy_team += ct_dict[current_hero][hero]
            hero_ave_ct_towards_enemy_team /= 5
    
        red_team_score += hero_ave_sy_towards_join_team * hero_ave_ct_towards_enemy_team

    if blue_team_score>= red_team_score:
        return 1
    else:
        return 0
    
# SC validation
def sc_validation(X, y, champ_row_name_dic):
    tp = 0
    total = 0
    for line in range(len(X)):
        v = X[line]
        r = y[line]
        
        # reformat X
        blue_team = []
        red_team = []
        for i in range(len(v)):
            if v[i] == 1 and i%2 == 0:
                blue_team.append(i//2)
            elif v[i] == 1:
                red_team.append((i-1)//2)
                
        if sv_predict(blue_team, red_team, champ_row_name_dic) == r:
            tp += 1
        total += 1
        
    return tp/total

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

    if fit_kNN_model:
        # model choice
        kNN = KNeighborsClassifier(n_neighbors=20, weights = 'uniform')
        kNN.fit(team, win)
        
        # save to pickle
        pickle.dump(kNN, open('saved_model/kNN_model.save', 'wb'))
    
        print('Model fit and saved!')
        
        return
    
    if fit_NB_model:
        # model choice
        NB = MultinomialNB()
        NB.fit(team, win)
        
        # save to pickle
        pickle.dump(NB, open('saved_model/NB_model.save', 'wb'))
    
        print('Model fit and saved!')
        
        return
    
    # reload model from pickle to save time
    kNN = pickle.load(open('saved_model/kNN_model.save', 'rb'))
    NB = pickle.load(open('saved_model/NB_model.save', 'rb'))
    print('Model reloaded!')
    
    # validation
    if run_valid:
        print('Validation started!')
        
        print('kNN: ', np.average(cross_validate(kNN, team_sample, win_sample, cv=5, scoring='accuracy')['test_score']))
        print('SC: ', sc_validation(team, win, champ_row_name_dic))
        print('NB: ', np.average(cross_validate(NB, team, win, cv=5, scoring='accuracy')['test_score']))
        
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
    
    print("-----------------------------------")
    # player as blue team 
    if team_side == 'b':
        # ban round
        # b3 ban
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        ban = champ_dic[random.choice(champ_pool)]
        blue_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print("Player B3 ban: "+ban)
        input("Press enter to continue...\n-----------------------------------")
        
        # r3 ban: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        ban = champ_dic[random.choice(champ_pool)]
        red_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print("Player R3 ban: "+ban)
        input("Press enter to continue...\n-----------------------------------")
        
        # b4 ban
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        ban = champ_dic[random.choice(champ_pool)]
        blue_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print("Player B4 ban: "+ban)
        input("Press enter to continue...\n-----------------------------------")
        
        # r4 ban: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        ban = champ_dic[random.choice(champ_pool)]
        red_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print("Player R4 ban: "+ban)
        input("Press enter to continue...\n-----------------------------------")
        
        # b5 ban
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        ban = champ_dic[random.choice(champ_pool)]
        blue_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print("Player B5 ban: "+ban)
        input("Press enter to continue...\n-----------------------------------")
        
        # r5 ban: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        ban = champ_dic[random.choice(champ_pool)]
        red_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print("Player R5 ban: "+ban)
        input("Press enter to continue...\n-----------------------------------")

        # pick round
        # b1 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # highest win rate champ for the first pick
        best = s_and_c_overall_best(champ_row_name_dic)
        print('Top 7 champions: ', best)
        for c in best:
            if champ_dic[c] in champ_pool:
                pick = c
                break
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("\nPlayer B1' recommended pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")

        # r1 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R1 pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # r2 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R2 pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # b2 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        sc_result = s_and_c_rs(blue_team, red_team, team_side, blue_ban, red_ban, champ_dic, champ_row_name_dic)
        NB_result = NB_rs(blue_team, red_team, team_side, champ_id, champ_pool, NB, win, champ_dic)
        print('kNN recommend list: ', kNN_result)
        print('\nSC recommend list: ', sc_result)
        print('\nNB recommend list: ', NB_result)
        pick = champ_decision(kNN_result, sc_result, NB_result)
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("\nPlayer B2' recommended pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")     
        
        # b3 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        sc_result = s_and_c_rs(blue_team, red_team, team_side, blue_ban, red_ban, champ_dic, champ_row_name_dic)
        NB_result = NB_rs(blue_team, red_team, team_side, champ_id, champ_pool, NB, win, champ_dic)
        print('kNN recommend list: ', kNN_result)
        print('\nSC recommend list: ', sc_result)
        print('\nNB recommend list: ', NB_result)
        pick = champ_decision(kNN_result, sc_result, NB_result)
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("\nPlayer B3' recommended pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # r3 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R3 pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # r4 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R4 pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # b4 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        sc_result = s_and_c_rs(blue_team, red_team, team_side, blue_ban, red_ban, champ_dic, champ_row_name_dic)
        NB_result = NB_rs(blue_team, red_team, team_side, champ_id, champ_pool, NB, win, champ_dic)
        print('kNN recommend list: ', kNN_result)
        print('\nSC recommend list: ', sc_result)
        print('\nNB recommend list: ', NB_result)
        pick = champ_decision(kNN_result, sc_result, NB_result)
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("\nPlayer B4' recommended pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # b5 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        sc_result = s_and_c_rs(blue_team, red_team, team_side, blue_ban, red_ban, champ_dic, champ_row_name_dic)
        NB_result = NB_rs(blue_team, red_team, team_side, champ_id, champ_pool, NB, win, champ_dic)
        print('kNN recommend list: ', kNN_result)
        print('\nSC recommend list: ', sc_result)
        print('\nNB recommend list: ', NB_result)
        pick = champ_decision(kNN_result, sc_result, NB_result)
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("\nPlayer B5' recommended pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # r5 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player R5 pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
    
    # player as red team    
    else:
        # ban round
        # b3 ban
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        ban = champ_dic[random.choice(champ_pool)]
        blue_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print("Player B3 ban: "+ban)
        input("Press enter to continue...\n-----------------------------------")
        
        # r3 ban: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        ban = champ_dic[random.choice(champ_pool)]
        red_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print("Player R3 ban: "+ban)
        input("Press enter to continue...\n-----------------------------------")
        
        # b4 ban
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        ban = champ_dic[random.choice(champ_pool)]
        blue_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print("Player B4 ban: "+ban)
        input("Press enter to continue...\n-----------------------------------")
        
        # r4 ban: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        ban = champ_dic[random.choice(champ_pool)]
        red_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print("Player R4 ban: "+ban)
        input("Press enter to continue...\n-----------------------------------")
        
        # b5 ban
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        ban = champ_dic[random.choice(champ_pool)]
        blue_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print("Player B5 ban: "+ban)
        input("Press enter to continue...\n-----------------------------------")
        
        # r5 ban: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        ban = champ_dic[random.choice(champ_pool)]
        red_ban.append(ban)
        champ_pool.remove(champ_dic[ban])
        print("Player R5 ban: "+ban)
        input("Press enter to continue...\n-----------------------------------")
        
        # pick round
        # b1 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B1 pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # r1 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        sc_result = s_and_c_rs(blue_team, red_team, team_side, blue_ban, red_ban, champ_dic, champ_row_name_dic)
        NB_result = NB_rs(blue_team, red_team, team_side, champ_id, champ_pool, NB, win, champ_dic)
        print('kNN recommend list: ', kNN_result)
        print('\nSC recommend list: ', sc_result)
        print('\nNB recommend list: ', NB_result)
        pick = champ_decision(kNN_result, sc_result, NB_result)
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("\nPlayer R1' recommended pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # r2 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        sc_result = s_and_c_rs(blue_team, red_team, team_side, blue_ban, red_ban, champ_dic, champ_row_name_dic)
        NB_result = NB_rs(blue_team, red_team, team_side, champ_id, champ_pool, NB, win, champ_dic)
        print('kNN recommend list: ', kNN_result)
        print('\nSC recommend list: ', sc_result)
        print('\nNB recommend list: ', NB_result)
        pick = champ_decision(kNN_result, sc_result, NB_result)
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("\nPlayer R2' recommended pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # b2 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B2 pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # b3 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B3 pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # r3 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        sc_result = s_and_c_rs(blue_team, red_team, team_side, blue_ban, red_ban, champ_dic, champ_row_name_dic)
        NB_result = NB_rs(blue_team, red_team, team_side, champ_id, champ_pool, NB, win, champ_dic)
        print('kNN recommend list: ', kNN_result)
        print('\nSC recommend list: ', sc_result)
        print('\nNB recommend list: ', NB_result)
        pick = champ_decision(kNN_result, sc_result, NB_result)
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("\nPlayer R3' recommended pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # r4 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        sc_result = s_and_c_rs(blue_team, red_team, team_side, blue_ban, red_ban, champ_dic, champ_row_name_dic)
        NB_result = NB_rs(blue_team, red_team, team_side, champ_id, champ_pool, NB, win, champ_dic)
        print('kNN recommend list: ', kNN_result)
        print('\nSC recommend list: ', sc_result)
        print('\nNB recommend list: ', NB_result)
        pick = champ_decision(kNN_result, sc_result, NB_result)
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("\nPlayer R4' recommended pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # b4 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B4 pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # b5 pick: random
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        pick = champ_dic[random.choice(champ_pool)]
        blue_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("Player B5 pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        
        # r5 pick
        print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)
        # rs algorithm
        kNN_result = kNN_rs(blue_team, red_team, team_side, champ_id, champ_pool, kNN, win, champ_dic)
        sc_result = s_and_c_rs(blue_team, red_team, team_side, blue_ban, red_ban, champ_dic, champ_row_name_dic)
        NB_result = NB_rs(blue_team, red_team, team_side, champ_id, champ_pool, NB, win, champ_dic)
        print('kNN recommend list: ', kNN_result)
        print('\nSC recommend list: ', sc_result)
        print('\nNB recommend list: ', NB_result)
        pick = champ_decision(kNN_result, sc_result, NB_result)
        red_team.append(champ_dic[pick])
        champ_pool.remove(champ_dic[pick])
        print("\nPlayer R5' recommended pick: "+pick)
        input("Press enter to continue...\n-----------------------------------")
        

    print_info(blue_team, red_team, blue_ban, red_ban, champ_dic)   
    print('Recommandation ended!')
    
main()
