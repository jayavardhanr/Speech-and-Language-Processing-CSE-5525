#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:49:35 2018

@author: jayavardhanreddy
"""
import itertools

#ListOfCards=['EIGHT','NINE','TEN','JACK','QUEEN']
#ListOfCards=['EIGHT','NINE','TEN']
ListOfCards=['EIGHT','NINE','TEN','JACK','QUEEN','KING','ACE']

state=3
threeOfAKind=1
fullHouse=2
with open("fullHouse_threeOfAKind.fst.txt","w+") as file:
    for index,baseCard in enumerate(ListOfCards):
        ##8999J                   
        cardsBefore=ListOfCards[:index]
        cardsAfter=ListOfCards[index+1:]     
        if len(cardsBefore)>0 and len(cardsAfter)>0:
            for firstCard in cardsBefore:
                for secondCard in cardsAfter:
                    file.write(str(0)+' '+str(state+1)+' '+firstCard+' '+'-'+'\n')
                    print(str(0)+' '+str(state+1)+' '+firstCard+' '+'-')
                    state+=1
                    for i in range(3):
                        file.write(str(state)+' '+str(state+1)+' '+baseCard+' '+'-'+'\n')
                        print(str(state)+' '+str(state+1)+' '+baseCard+' '+'-')
                        state+=1
                        
                    file.write(str(state)+' '+str(threeOfAKind)+' '+secondCard+' '+'THREE-OF-A-KIND'+'\n')       
                    print(str(state)+' '+str(threeOfAKind)+' '+secondCard+' '+'THREE-OF-A-KIND')
                    state+=1
        
        ##8889J
        if len(ListOfCards[index+1:])>=1:
            for i in range(3):
                if i==0:
                    file.write(str(0)+' '+str(state)+' '+baseCard+' '+'-'+'\n')
                    print(str(0)+' '+str(state)+' '+baseCard+' '+'-')
                else:
                    file.write(str(state)+' '+str(state+1)+' '+baseCard+' '+'-'+'\n')
                    print(str(state)+' '+str(state+1)+' '+baseCard+' '+'-')
                    state+=1
                    
            othercards=ListOfCards[index+1:]    
           
            interState=state
            if len(othercards)>=2:
                for firstCard,secondCard in itertools.combinations(othercards, 2):
                    file.write(str(interState)+' '+str(state+1)+' '+firstCard+' '+'-'+'\n')
                    print(str(interState)+' '+str(state+1)+' '+firstCard+' '+'-')
                    state+=1
                    file.write(str(state)+' '+str(threeOfAKind)+' '+secondCard+' '+'THREE-OF-A-KIND'+'\n')
                    print(str(state)+' '+str(threeOfAKind)+' '+secondCard+' '+'THREE-OF-A-KIND')
            #88899
            if len(othercards)>=1:
                for firstCard in othercards:
                    file.write(str(interState)+' '+str(state+1)+' '+firstCard+' '+'-'+'\n')
                    print(str(interState)+' '+str(state+1)+' '+firstCard+' '+'-')
                    state+=1
                    file.write(str(state)+' '+str(fullHouse)+' '+firstCard+' '+'FULL-HOUSE'+'\n')
                    print(str(state)+' '+str(fullHouse)+' '+firstCard+' '+'FULL-HOUSE')
            
        ##89JJJ   
        if len(ListOfCards[:index])>=1:
            othercards=ListOfCards[:index] 
            
            if len(othercards)>=2:
                for firstCard,secondCard in itertools.combinations(othercards, 2):
                    file.write(str(0)+' '+str(state+1)+' '+firstCard+' '+'-'+'\n')
                    print(str(0)+' '+str(state+1)+' '+firstCard+' '+'-')
                    state+=1
                    file.write(str(state)+' '+str(state+1)+' '+secondCard+' '+'-'+'\n')
                    print(str(state)+' '+str(state+1)+' '+secondCard+' '+'-')
                    state+=1
                    
                    for i in range(3):
                        if i==2:
                            file.write(str(state)+' '+str(threeOfAKind)+' '+baseCard+' '+'THREE-OF-A-KIND'+'\n')
                            print(str(state)+' '+str(threeOfAKind)+' '+baseCard+' '+'THREE-OF-A-KIND')
                        else:
                            file.write(str(state)+' '+str(state+1)+' '+baseCard+' '+'-'+'\n')
                            print(str(state)+' '+str(state+1)+' '+baseCard+' '+'-')
                            state+=1
            #88999
            if len(othercards)>=1:
                for firstCard in othercards:
                    file.write(str(0)+' '+str(state+1)+' '+firstCard+' '+'-'+'\n')
                    print(str(0)+' '+str(state+1)+' '+firstCard+' '+'-')
                    state+=1
                    file.write(str(state)+' '+str(state+1)+' '+firstCard+' '+'-'+'\n')
                    print(str(state)+' '+str(state+1)+' '+firstCard+' '+'-')
                    state+=1
                    for i in range(3):
                        if i==2:
                            file.write(str(state)+' '+str(fullHouse)+' '+baseCard+' '+'FULL-HOUSE'+'\n')
                            print(str(state)+' '+str(fullHouse)+' '+baseCard+' '+'FULL-HOUSE')
                        else:
                            file.write(str(state)+' '+str(state+1)+' '+baseCard+' '+'-'+'\n')
                            print(str(state)+' '+str(state+1)+' '+baseCard+' '+'-')
                            state+=1
    file.write('1\n')
    file.write('2\n')
     
                 
                 
                        
    
       
        
                