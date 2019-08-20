import Basic_DQN_for_Power_Maximization_2 as Basic_DQN
import time
import torch #import  torch
import torch.nn as nn #import nn
import numpy as np
import matplotlib.pyplot as plt
import random


Torque=np.float(0)
RPM = np.float(383)
Wind_Speed=np.float(6.4)
Power=np.float(0)
reward=Power
state=[[Power,Torque,Wind_Speed]]
state_last = state
counter=1
action_preformed=0
Power_rec=[0]
Count_rec=[0]
Torque_Max = 0.2576
RPM_max = (-1251*np.power(Torque_Max,2)-260*Torque_Max+383)*Wind_Speed/6.4
MaxPower =[RPM_max * Torque_Max]

got_reward=False

while True:
    print('Power:',Power,', Torque:',Torque,', RPM:', RPM,', Reward:', reward, ', Wind:',Wind_Speed)
    action_preformed = Basic_DQN.Learn(state,action_preformed,state_last,reward,counter)
    #np.float(action_preformed)
    if action_preformed.item() == 0:
        Torque=Torque - np.float(0.001)
    elif action_preformed.item() == 1:
        Torque = Torque
    elif action_preformed.item() == 2:
        Torque = Torque + np.float(0.001)
    elif action_preformed.item() == 3:
        Torque = Torque - np.float(0.01)
    elif action_preformed.item() == 4:
        Torque = Torque + np.float(0.01)


    if Torque < 0:
        Torque = np.float(0)
        reward = np.float(-2)
        got_reward=True
    elif Torque > 0.5:
        Torque = np.float(0.5)
        reward = np.float(-2)
        got_reward = True



    RPM = (-1251*np.power(Torque,2)-260*Torque+383)*Wind_Speed/6.4

    if counter > 200 and counter%5:
        RPM=RPM+np.float((random.random()-0.5)*0.2)

    # if counter > 3000 and counter % 400 == 0:
    #     Wind_Speed = Wind_Speed+np.float((random.random()-0.5)*3)
    # if counter > 3000:
    #     Wind_Speed = Wind_Speed + (np.sin((counter - 3000) / 500))/1000

        


    # if counter > 2000:# and  counter%500==0:
    #     Wind_Speed=6.4+np.sin((counter-2000)/500)*4#np.float((random.random()-0.5)*3)
    #     np.sin(counter/4000)


    if RPM<0:
        RPM=0
        reward=np.float(-2)
        got_reward = True

    RPM=np.float(RPM)
    Power_New = RPM*Torque

    if Power_New - Power > 0 and not got_reward:
        reward = np.float(+0.1+Power_New/200)
    elif Power_New - Power == 0 and not got_reward:
        reward = np.float(+Power_New/100)
    elif Power_New - Power < 0 and not got_reward:
        reward = np.float(-0.2+Power_New/200)
    got_reward=False #reset got reward
    Power=Power_New
    #time.sleep(0.1)
    state_last=state
    state=[[Power,Torque,Wind_Speed]]
    Power_rec.append(Power)
    Count_rec.append(counter)
    counter = counter+1
    Torque_Max = 0.2576
    RPM_max = (-1251*np.power(Torque_Max,2)-260*Torque_Max+383)*Wind_Speed/6.4
    Power_Max =RPM_max * Torque_Max
    MaxPower.append(Power_Max)

    if counter % 4000 == 0:
        plt.plot(Count_rec,Power_rec,Count_rec,MaxPower)
        plt.show()


