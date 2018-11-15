# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:29:32 2018

@author: Karas
"""

import pandas as pd

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
    # read ori_data
    champ_df = pd.read_csv(
        filepath_or_buffer='ori_data/champs.csv', 
        header=0,
        sep=',')

    version_df = pd.read_csv(
        filepath_or_buffer='ori_data/matches.csv', 
        header=0,
        sep=',')

    team_df = pd.read_csv(
        filepath_or_buffer='ori_data/participants.csv', 
        header=0,
        sep=',')

    result_df = pd.read_csv(
        filepath_or_buffer='pre_data/stats_all.csv', 
        header=0,
        sep=',')
   
    # champ_id list
    champ_id = champ_df.id.tolist()
    
    # reformat into new csv    
    # set col_name
    col_name=[]
    for i in range(0,138):
        col_name.append('b'+str(i))
        col_name.append('r'+str(i))
    col_name.append('b_win')
    
    # print col_name
    file = open('pre_data/s8_champ_win_data.csv','w')
    s = str(col_name).replace(" ", "")
    s = s.replace("'", "")
    s = s[1:len(s)-1]
    file.write(s+'\n')
    
    team_index = 0
    result_index = 0
    for index, row in version_df.iterrows():
        match_id = row.id
        # only keep season 8
        if row.seasonid == 8:
            # skip unused row in team_df
            while (team_df[team_index:team_index+1].matchid.values[0]<match_id):
                team_index+=1
            # skip player number < 10
            if team_df[team_index+9:team_index+10].matchid.values[0]!= match_id:
                continue
            # skip unused row in result_df
            while result_df[result_index:result_index+1].id.values[0]<team_df[team_index:team_index+1].id.values[0]:
                result_index+=1

            # get team info
            blue_team = []
            red_team = []
            while team_df[team_index:team_index+1].matchid.values[0]==match_id:
                #blue team
                if team_df[team_index:team_index+1].player.values[0]<=5:
                    blue_team.append(team_df[team_index:team_index+1].championid.values[0])
                #red team
                else:
                    red_team.append(team_df[team_index:team_index+1].championid.values[0])               
                team_index+=1
            
            # print to file, format: champ_one_hot,result
            info = champ_one_hot(blue_team,champ_id,'b',[])
            info = champ_one_hot(red_team,champ_id,'r',info)
            info.append(result_df[result_index:result_index+1].win.values[0])
            s = str(info).replace(" ", "")
            s = s[1:len(s)-1]
            file.write(s+'\n')
            
    file.close()
    print("Output complete!")
    
main()
