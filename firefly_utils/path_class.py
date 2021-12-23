#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:20:14 2020

@author: edoardo
get path script

"""
from python_monkey_info import monkey_info_class

class get_paths_class(object):
    def __init__(self):
        # path to allen institute code
        self.ecephys_spike_sorting = '/Users/edoardo/Work/Code/ecephys_spike_sorting/'
        # path to npz local data folder
        self.local_concat = '/Volumes/WD Edo/firefly_analysis/LFP_band/DATASET/PPC+PFC+MST/'
        # path to were data are stored in the cluster diveded by array and matrix
        self.cluster_data_basepath = '/Volumes/server/Data/Monkey2_newzdrive/%s/neural data/Sorted'
        self.cluster_data_basepath_array = '/Volumes/server/Data/Monkey2_newzdrive/%s/neural data/PLEXON FILES/Sorted/'
        
        # path to the monkey info folder
        monkey_info_path = '/Users/edoardo/Work/Code/Firefly_sina/firefly-monkey/'
        # path to sorted
        self.monkey_info = monkey_info_class(monkey_info_path)
        
        # hpc data dir
        self.data_hpc = '/scratch/jpn5/dataset_firefly/'
        
        # hpc code dir
        self.code_hpc = '/scratch/jpn5/final_gam_fit_coupling/'
        
    def get_path(self, path_type, session=None):
        if path_type == 'ecephys_spike_sorting':
            return self.ecephys_spike_sorting
        elif path_type == 'local_concat':
            return self.local_concat
        elif path_type == 'cluster_array_data':
            sorted_fold = self.cluster_data_basepath_array % self.monkey_info.get_folder(session)
            return sorted_fold
        elif path_type == 'cluster_data':
            sorted_fold = self.cluster_data_basepath % self.monkey_info.get_folder(session)
            return sorted_fold
        elif path_type == 'data_hpc':
            return self.data_hpc
        elif path_type == 'code_hpc':
            return self.code_hpc
        
        
       
        