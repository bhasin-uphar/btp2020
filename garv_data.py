import csv
import sys
import os
from shutil import copy

path='E://BTP_2020//5552743//' #path of csv file
csv_path='garv_data_bihar.csv'
l={}
with open(path + csv_path,'r') as f:
    reader=csv.reader(f, delimiter=',')
    for row in reader:
        l[row[0]]=row[9]
        
path1='E://BTP_2020//5552743//IndianVillagesDataset_30m//imagery_res30_48bands' #path of all photos
path2='E://BTP_2020//5552743//electrified' #path for electrified
path3='E://BTP_2020//5552743//unelectrified' #path for unelectrified

dirs = os.listdir(path1) 

for i in dirs:
    try:
        if l[i.split('.')[0].split('-')[-1]]=="1":
            copy(path1+'//'+i,path2)
        else:
            copy(path1+'//'+i,path3)
    except:
        pass

print('done')
