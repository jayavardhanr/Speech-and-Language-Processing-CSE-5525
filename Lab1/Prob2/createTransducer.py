#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:59:15 2018

@author: jayavardhanreddy
"""
listPhones=['ah','ao','ax','ay','eh','ey','f','ih','iy','k','n','ow','r','s','SIL','t','th','uw','v','w','z']
#listPhones=['eh','ao','ax']

with open("dur.fst.txt","w+") as file:
    # Use file to refer to the file object
    for i in range(len(listPhones)):
        
        '''
        print(str(0)+' '+str(3*i+1)+' '+listPhones[i]+' -')
        print(str(3*i+1)+' '+str(3*i+2)+' '+listPhones[i]+' -')
        print(str(3*i+2)+' '+str(3*i+3)+' '+listPhones[i]+' '+listPhones[i])
        print(str(3*i+3)+' '+str(3*i+3)+' '+listPhones[i]+' -')
        '''
       
        file.write(str(0)+' '+str(3*i+1)+' '+listPhones[i]+' -\n')
        file.write(str(3*i+1)+' '+str(3*i+2)+' '+listPhones[i]+' -\n')
        file.write(str(3*i+2)+' '+str(3*i+3)+' '+listPhones[i]+' '+listPhones[i]+'\n')
        file.write(str(3*i+3)+' '+str(3*i+3)+' '+listPhones[i]+' -'+'\n')
        for j in range(len(listPhones)):
            if i!=j:
                '''
                #print(str(3*i+1)+' '+str(3*j+1)+' '+listPhones[j]+' -')
                #print(str(3*i+2)+' '+str(3*j+1)+' '+listPhones[j]+' -')
                print(str(3*i+3)+' '+str(3*j+1)+' '+listPhones[j]+' -')
                '''
               
                #file.write(str(3*i+1)+' '+str(3*j+1)+' '+listPhones[j]+' -'+'\n')
                #file.write(str(3*i+2)+' '+str(3*j+1)+' '+listPhones[j]+' -'+'\n')
                file.write(str(3*i+3)+' '+str(3*j+1)+' '+listPhones[j]+' -'+'\n')
              
    for k in range(64):
        file.write(str(k)+'\n')
               
with open("words.voc") as file:
    message = file.read()
    print(message)
   