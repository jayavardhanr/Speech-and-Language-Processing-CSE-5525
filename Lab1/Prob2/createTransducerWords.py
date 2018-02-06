#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 23:57:15 2018

@author: jayavardhanreddy
"""
with open('dict.fst.txt','w+') as newfile:
    with open("dict.txt","r+") as file:
        i=0
        print(str(0)+' '+str(0)+' '+'SIL'+' '+'-')
        newfile.write(str(0)+' '+str(0)+' '+'SIL'+' '+'-'+'\n')
        for line in file:
            split=line.split()
            word=split[0]
            phones=split[1:]
            print(str(0)+' '+str(i+1)+' '+phones[0]+' '+'-')
            newfile.write(str(0)+' '+str(i+1)+' '+phones[0]+' '+'-'+'\n')
            i+=1
            for phone in phones[1:-1]:
                print(str(i)+' '+str(i+1)+' '+phone+' '+'-')
                newfile.write(str(i)+' '+str(i+1)+' '+phone+' '+'-'+'\n')
                i+=1
            print(str(i)+' '+str(i+1)+' '+phone+' '+word)
            newfile.write(str(i)+' '+str(i+1)+' '+phone+' '+word+'\n')
            i+=1
            print(str(i)+' '+str(0)+' '+'SIL'+' '+'-')
            newfile.write(str(i)+' '+str(0)+' '+'SIL'+' '+'-'+'\n')
    
