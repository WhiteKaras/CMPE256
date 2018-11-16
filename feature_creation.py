import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create Heros' Synergy and Counter scores to return top 5 choices
def create_feature():
    """
    Use preprocessed data to compute each heroes' Synergy and Counter scores against all other heroes
    """

    data = pd.read_csv('./pre_data/s8_chap_win_data.csv')
    logger.info("This Data has %d matches", len(data))

    logger.info("Computing hero abilities map...")
    make_map(data)


def make_map(df):
    """
    Calculate synergies and counters dictionary
    """
    ct = dict()
    sy = dict()

    # matches number against other heroes
    # wins number against other heroes
    # wining rate against other heroes

    ct['wins'] = np.zeros((138, 138))
    ct['matches'] = np.zeros((138, 138))
    ct['winrate'] = np.zeros((138, 138))

    sy['wins'] = np.zeros((138, 138))
    sy['matches'] = np.zeros((138, 138))
    sy['winrate'] = np.zeros((138, 138))

    for v in df.values:
        add_match(sy, ct, v)


def add_match(synergy, counter, match):
    """
    add each match's information into synergy and counter dict
    """
    print 'TODO'


if __name__ == '__main__':
    create_feature()
