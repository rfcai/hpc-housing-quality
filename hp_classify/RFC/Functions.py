#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 14:55:44 2018

@author: kevinhsu
"""

import pandas as pd
import timeit
import numpy as np


def Load_Data(file_name):
    """Load Csv to dataframe"""    
    start = timeit.default_timer()
    df = pd.read_csv(file_name,low_memory=False) 
    attr = ['int_year',
            'housing_roof_num','housing_wall_num','housing_floor_num','iso3']

    df = df[attr]
    df = df[df['housing_wall_num']!=0]
    df = df[df['housing_roof_num']!=0] 
    df = df[df['housing_floor_num']!=0] 

    #df = df[df['housing_roof_num']!=0 or df['housing_roof_num']!= None]
    stop = timeit.default_timer()
    print("Runtime:{:.2f} sec".format(stop - start))
    
    return df


def Cleaning_Block(df, word):
    """
    filter out special characters within number columns, 
    and convert numbers from string to int
    """ 
    df['housing_'+word+'_num'] = df['housing_'+word+'_num'].replace('?',None)
    df['housing_'+word+'_num'] = df['housing_'+word+'_num'].replace('.',None)
    df['housing_'+word+'_num'] = pd.to_numeric(df['housing_'+word+'_num'])
    
def Ranking(df, vals):
    """
    Then keep values within [10,35], cause the values out of this set is considered as missing values
    rank all three categories
    """
    for val in vals:
        Cleaning_Block(df, val)
        df = df[(df['housing_'+val+'_num']<=35) & (df['housing_'+val+'_num']>=10)]
        df[val+'_rank'] = (df['housing_'+val+'_num']/10).apply(np.floor)
        df.loc[df[val+'_rank'] == 0, val+'_rank'] = 0
        df.loc[df[val+'_rank'] >= 4, val+'_rank'] = 4
    return df

#Still working on differnet kernel methods
def Add_Kernel(df):
    """feature enigeering using kernel method"""
    df['kernel1'] = df['housing_wall_num'] * df['housing_floor_num']
    
def Shuffle_Redistribute(df, LABEL, Redistribute=True):
    """Greneralize the data set distribution of the classes, Shuffle the data set to unbiase"""
    if Redistribute:
        rank1 = df[df[LABEL+'_rank']==1]
        rank2 = df[df[LABEL+'_rank']==2]
        rank3 = df[df[LABEL+'_rank']==3]
        
        list_for_rank_df = [rank1, rank2, rank3]
        new_list = []
        base = min(len(rank1),len(rank2),len(rank3))
        for rank in list_for_rank_df:
            if len(rank) != base:
                rank['rand_num'] = np.random.uniform(0,1, len(rank)) 
                rank = rank[rank['rand_num']<= base/len(rank)]
                rank = rank.drop(columns = 'rand_num')
            new_list.append(rank)
        #rank3['rand_num'] = np.random.uniform(0,1, len(rank3)) 
        #rank3 = rank3[rank3['rand_num']<= base/len(rank3)]
        #rank3 = rank3.drop(columns = 'rand_num')

        df = pd.concat(new_list)
        
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def Train_Test_Split(df):
    """split data to test and training set"""
    df['train'] = np.random.uniform(0, 1, len(df)) <= .75
    df['iso3'] = pd.factorize(df['iso3'])[0]
    train = df[df['train']==True]
    test = df[df['train']==False]
    return train, test
    
    