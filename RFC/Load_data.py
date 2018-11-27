#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:55:25 2018

@author: kevinhsu
"""


import pandas as pd
import timeit



def Load_Data(file_name):
    """Load Csv to dataframe"""    
    start = timeit.default_timer()
    df = pd.read_csv(file_name) 
    attr = ['int_year',
            'housing_roof_num','housing_wall_num','housing_floor_num','iso3']

    df = df[attr]
    df = df[(df['housing_wall_num']!=0)] 
    df = df[(df['housing_roof_num']!=0)] 
    df = df[(df['housing_floor_num']!=0)] 
    #df = df[df['housing_roof_num']!=0 or df['housing_roof_num']!= None]
    stop = timeit.default_timer()
    print(stop - start)
    return df


def Cleaning_Block(df, word):
    """clean special characters"""   #better to use regular expression
    df['housing_'+word+'_num'] = df['housing_'+word+'_num'].replace('?',0)
    df['housing_'+word+'_num'] = df['housing_'+word+'_num'].replace('.',0)
    df['housing_'+word+'_num'] = pd.to_numeric(df['housing_'+word+'_num'])
    
    
