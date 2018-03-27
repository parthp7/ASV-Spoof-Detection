#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 19:38:49 2018

@author: parth
"""

DIR_PATH = "/media/parth/Entertainment/ASV2015/"


def file_read(filename):
    
    wav_files = list()
    
    with open("{}{}".format(DIR_PATH, filename), "r") as file:
        for line in file:
            if line == "": break
            record = line.split()
            if record[0] == "T1" and record[3] == "human":
                wav_files.append(record)
    
    return wav_files





if __name__ == "__main__":
    
    wav_files = file_read("CM_protocol/cm_train.trn")
    