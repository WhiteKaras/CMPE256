import pandas as pd
import numpy as np
import logging, math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create Heros' Synergy and Counter scores to return top 5 choices
def create_features():
    """
    Use preprocessed data to compute each hero's Synergy and Counter scores against all other heroes
    """
    pre_data = pd.read_csv('./pre_data/s8_champ_win_data.csv')

    logger.info("This data et has %d matches", len(pre_data))
    logger.info("Computing hero synergy, counter, self_win_rate...")

    generate_maps(pre_data)


def generate_maps(data):
    """
    Compute synergy and counter dict and transform to matrix, then save to csv file
    """
    counter = dict()
    synergy = dict()
    self_win = dict()

    # each hero has:
    # total wins number against other 137 heroes
    # total matches number against other 137 heroes
    # total wining rate against other 137 heroes

    counter['wins'] = np.zeros((138, 138))
    counter['matches'] = np.zeros((138, 138))
    counter['winrate'] = np.zeros((138, 138))

    synergy['wins'] = np.zeros((138, 138))
    synergy['matches'] = np.zeros((138, 138))
    synergy['winrate'] = np.zeros((138, 138))

    self_win['wins'] = np.zeros(138)
    self_win['matches'] = np.zeros(138)
    self_win['winrate'] = np.zeros(138)

    for instance in data.values:
        build_dict(synergy, counter, self_win, instance)

    compute_wining_rates(synergy, counter, self_win, 138)
    sy_mat, ct_mat, s_win_mat = build_matrix(synergy, counter, self_win, 138)

    np.savetxt('./fea_data/heroes_synergies_readable.csv', sy_mat)
    np.savetxt('./fea_data/heroes_counters_readable.csv', ct_mat)
    np.savetxt('./fea_data/heroes_self_win_rate_readable.csv', s_win_mat)
    np.save('./fea_data/heroes_synergies', sy_mat)
    np.save('./fea_data/heroes_counters', ct_mat)
    np.save('./fea_data/heroes_self_win_rate', s_win_mat)

    logger.info("Synergy and Counter map created successfully!")


def build_dict(synergy, counter, self_win, match):
    """
    fulfill synergy and counter dict with each historical match record
    """
    b_win = match[276]

    b_team = []
    r_team = []

    # go through each match record and find blue team heroes and red team heroes
    for i in range(len(match)-1):
        if match[i] == 1:
            if i % 2 == 1:
                r_team.append(math.floor(i/2))
            else:
                b_team.append(int(i/2))

    # skip dirty data that does not match a 5 vs 5 situation
    if len(r_team) == 5 and len(b_team) == 5:
        # go through a 5 vs 5 regular match of each hero team up/against each hero
        for i in range(5):
            # for self win
            self_win['matches'][b_team[i]] += 1
            self_win['matches'][r_team[i]] += 1

            if b_win == 1:
                self_win['wins'][b_team[i]] += 1
            else:
                self_win['wins'][r_team[i]] += 1

            for j in range(5):
                b_hero_i = b_team[i]
                b_hero_j = b_team[j]
                r_hero_i = r_team[i]
                r_hero_j = r_team[j]

                # for synergy
                if i != j:
                    synergy['matches'][r_hero_i, r_hero_j] += 1
                    synergy['matches'][b_hero_i, b_hero_j] += 1

                    if b_win == 1:
                        synergy['wins'][b_hero_i, r_hero_j] += 1
                    else:
                        synergy['wins'][r_hero_i, b_hero_j] += 1

                # for counter
                counter['matches'][r_hero_i, b_hero_j] += 1
                counter['matches'][b_hero_i, r_hero_j] += 1

                if b_win == 1:
                    counter['wins'][b_hero_i, r_hero_j] += 1
                else:
                    counter['wins'][r_hero_i, b_hero_j] += 1


def compute_wining_rates(synergy, counter, self_win, heroes_num):
    """
    Loop through synergy and counter dict and compute winning rate
    """
    for i in range(heroes_num):
        self_win['winrate'][i] = self_win['wins'][i] / self_win['matches'][i] if self_win['matches'][i] != 0.0 else 0.0
        for j in range(heroes_num):
            if i != j:
                if synergy['matches'][i, j] != 0:
                    synergy['winrate'][i, j] = synergy['wins'][i, j] / float(synergy['matches'][i, j])

                if counter['matches'][i, j] != 0:
                    counter['winrate'][i, j] = counter['wins'][i, j] / float(counter['matches'][i, j])


def build_matrix(synergy, counter, self_win, heroes_num):
    """
    Loop through all combination of heroes to build score matrix for synergy and counter
    """
    sy_mat = np.zeros((heroes_num, heroes_num))
    ct_mat = np.zeros((heroes_num, heroes_num))
    s_win_mat = np.zeros(heroes_num)

    for i in range(heroes_num):
        s_win_mat[i] = self_win['winrate'][i]
        for j in range(heroes_num):
            if i != j:
                if synergy['matches'][i, j] > 0:
                    sy_mat[i, j] = synergy['winrate'][i, j]/(self_win['winrate'][i]+self_win['winrate'][j])
                else:
                    sy_mat[i, j] = 0

                if counter['matches'][i, j] > 0:
                    ct_mat[i, j] = counter['winrate'][i, j]/(self_win['winrate'][i]+1-self_win['winrate'][j])
                else:
                    ct_mat[i, j] = 0

    return sy_mat, ct_mat, s_win_mat


if __name__ == '__main__':
    create_features()
