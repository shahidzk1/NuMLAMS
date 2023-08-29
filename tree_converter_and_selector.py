import numpy as np
import pandas as pd
import uproot
from concurrent.futures import ThreadPoolExecutor

""""
This class imports a root tree file and can convert to a dataframe. It can also apply selections on the data and store it as a dataframe

Args:
    file_path_name (string)   : The address of the root file and its name
    tree_name      (string)   : The name of the TTree object inside the file
    n_jobs         (int)      : The number of parallel jobs to run    
"""
class TreeHandler:
    def __init__(self, file_path_name, tree_name, n_jobs):
        self . file_path_name                           = file_path_name
        self . tree_name                                = tree_name
        self . n_jobs                                   = n_jobs
        self . executor                                   = ThreadPoolExecutor(n_jobs)
        
    def file_creator(self):
        """"
        Imports a root file into python
        Args: 
            None
        
        Returns
            file (uproot.models.TTree): An uproot object that contains the file and its tree
            
        """
        file = uproot.open(f"{self.file_path_name}:{self.tree_name}",decompression_executor=self.executor,interpretation_executor=self.executor)
        return file
        
    def get_dataframe(self):
        """"
        Converts the tree in the file to a dataframe
        
        Args: 
            None
        
        Returns
            df (Pandas.Dataframe) : The TTree converted into a dataframe
        """
        df = self.file_creator().arrays(library="pd",decompression_executor=self.executor,interpretation_executor=self.executor)
        return df
    
    def selection_bounds_based(self, variable_name, lower_bound, upper_bound):
        """"
        Returns data from dataframe by applying selection on a variable

        Args:
            df (Pandas.Dataframe)   : The input dataframe
            variable_in_df (string) : a column name in df
            lower_bound (float)     : the lower bound on the variable
            upper_bound (float)     : the upper bound on the variable

        Returns
            selec_df (Pandas.Dataframe): A subset of df after the application of selection

        """
        df = self . get_dataframe()
        df = df[(df[variable_name]>=lower_bound) & (df[variable_name]<upper_bound)]
        return df
    
    def fragmentation_selection(self, layer_info, frag_option, variable_name, lower_bound, upper_bound, df_option=None):
        '''
        The function applies the MC selection on detector parts to check whether a nuclei has frgmented or not

        Args:
            df (Pandas.Dataframe)   : The input dataframe
            layer_info (int)        : The number of layers to check for
            frag_option (string)    : fragmented or non-fragmented options

        Returns
            df (Pandas.Dataframe)   : A subset of df after the application of selection

        use case:
            df = non_fragmented(df, 4, 'fragmented')

            13 means upto layer 8 of the tracking

        '''

        var = ['mevmom1[0]','mevmom1[1]','mevmom1[2]','mevmom1[3]','mevmom1[4]','mevmom1[5]',
                  'mevmom1[6]','mevmom1[7]','mevmom1[8]','mevmom1[9]','mevmom1[10]','mevmom1[11]',
                  'mevmom1[12]','mevmom1[13]','mevmom1[14]','mevmom1[15]','mevmom1[16]',
                  'mevmom1[17]','mevmom1[18]','mevmom1[19]','mevmom1[20]']
        if df_option == "no_sel":
            df = self . get_dataframe()
        else:
            df = self . selection_bounds_based(variable_name, lower_bound, upper_bound)

        if frag_option=='non-fragmented':
            df = df[df[var[layer_info]]>0]
            return df

        if frag_option=='fragmented':
            df = df[df[var[layer_info]]<0]
            return df
            
    def labelled(self, label_frag, label_non_frag, layer_info, frag_option, variable_name, lower_bound, upper_bound, df_option=None):
        '''
        The function applies the fragmentation_selection function and labels the data

        Args:
            df (Pandas.Dataframe)   : The input dataframe
            label_frag (int)        : The label for the fragmented data
            label_non_frag (int)    : The label for the non-fragmented data

        Returns
            df_frag (Pandas.Dataframe)   : A subset of df after the application of fragmentation_selection with label "label_frag"
            df_non_frag (Pandas.Dataframe): A subset of df after the application of fragmentation_selection with label                                                        "label_non_frag"

        use case:
            df_frag, df_non_frag = non_fragmented(df, 0, 1)

        '''
        df_frag = self.fragmentation_selection(layer_info, 'fragmented', variable_name, lower_bound, upper_bound, df_option=None)
        df_non_frag = self.fragmentation_selection( layer_info, 'non-fragmented', variable_name, lower_bound, upper_bound, df_option=None)
        df_frag['label']= np.zeros(df_frag.shape[0])
        df_frag['label'] = df_frag['label'].replace(0,label_frag)
        df_non_frag['label']= np.zeros(df_non_frag.shape[0])
        df_non_frag['label'] = df_non_frag['label'].replace(0,label_non_frag)
        return df_frag, df_non_frag

