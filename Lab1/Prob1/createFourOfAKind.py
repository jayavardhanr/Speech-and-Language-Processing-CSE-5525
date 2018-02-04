#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:49:35 2018

@author: jayavardhanreddy
"""

ListOfCards=['EIGHT','NINE','TEN','JACK','QUEEN','KING','ACE']
#ListOfCards=['EIGHT','NINE','TEN']

state=2
finalState=1
with open("fourOfAkind.fst.txt","w+") as file:
    for index,baseCard in enumerate(ListOfCards):
        if len(ListOfCards[index+1:])>0:
            for i in range(4):
                if i==0:
                    file.write(str(0)+' '+str(state)+' '+baseCard+' '+'-'+'\n')
                    print(str(0)+' '+str(state)+' '+baseCard+' '+'-')
                else:
                    file.write(str(state)+' '+str(state+1)+' '+baseCard+' '+'-'+'\n')
                    print(str(state)+' '+str(state+1)+' '+baseCard+' '+'-')
                    state+=1
                            
            for card in ListOfCards[index+1:]:
                file.write(str(state)+' '+str(finalState)+' '+card+' '+'FOUR-OF-A-KIND'+'\n')
                print(str(state)+' '+str(finalState)+' '+card+' '+'FOUR-OF-A-KIND')
            state+=1
        
        if len(ListOfCards[:index])>0:          
            for card in ListOfCards[:index]:
                file.write(str(0)+' '+str(state)+' '+card+' '+'-'+'\n')
                print(str(0)+' '+str(state)+' '+card+' '+'-')
            
            for i in range(4):
                if i==3:
                    file.write(str(state)+' '+str(finalState)+' '+baseCard+' '+'FOUR-OF-A-KIND'+'\n')
                    print(str(state)+' '+str(finalState)+' '+baseCard+' '+'FOUR-OF-A-KIND')
                else:
                    file.write(str(state)+' '+str(state+1)+' '+baseCard+' '+'-'+'\n')
                    print(str(state)+' '+str(state+1)+' '+baseCard+' '+'-')
                state+=1
        
    file.write('1\n')
                