#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 20:50:07 2021

@author: ulachen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib as mpl
import math 
import schedule
import time 

from matplotlib.pyplot import figure
mpl.rcParams['mathtext.fontset'] = 'dejavusans' #'cm'
mpl.rcParams['font.family'] = 'sans-serif' #'STIXGeneral' #'cmu serif'
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams["figure.autolayout"] = True

#%% Functions 

def pos_in_box(pos, lbox):
    return (pos+lbox/2.) % lbox - lbox/2.

def displacement_table(coordinates, L):
    
    N=len(coordinates)
    r=np.zeros((N,N,3))
    for i in range(N): 
        for j in range(N): 
            r[i,j]=coordinates[i]-coordinates[j]
            
    r[r>=L]=r[r>=L]%L
    r[r<L]=r[r<L]%L-L
    r[r>=L/2]=r[r>=L/2]-L
    r[r<-L/2]=r[r<-L/2]+L
    
    disp=r
    
    return disp 

def my_pair_correlation(dists, natom, nbins, dr, lbox):

    histogram = np.histogram(dists, bins=nbins, range=(0, nbins*dr))
    r = (histogram[1] + dr/2)[:-1] # centers of the bins
    
    hist_ideal=np.zeros(len(r))
    
    for i in range(len(r)): 
        hist_ideal[i]=4*np.pi*((r[i]+dr/2)**3-(r[i]-dr/2)**3)*natom/(lbox**3)/3
    
    g=(np.divide(histogram[0],hist_ideal))/((natom-1)/2)
    return g, r

def my_legal_kvecs(maxn, lbox):

    kvecs = []
    for i in range(maxn+1): 
        for j in range(maxn+1): 
            for k in range(maxn+1): 
                kvecs.append(2*np.pi/lbox*np.array([i,j,k]))
    
    return np.array(kvecs)

def my_calc_rhok(kvecs, pos):

    rho=np.zeros(np.shape(kvecs)[0])
    rho=rho+0*1j
    for i in range(np.shape(kvecs)[0]): 
        kr=np.matmul(pos,kvecs[i])
        rho[i]=np.sum(np.exp(-kr*1j))
    return rho 

def my_calc_sk(kvecs, pos):

    s=1/(np.shape(pos)[0])*(np.absolute(my_calc_rhok(kvecs, pos)))**2
    return s

def data_col(file,i): 
    data = pd.read_csv(file, header = None, error_bad_lines=False)   
    df = pd.DataFrame(data)
    col=df.iloc[:,i]
    return col 

def data_excel(file,sheet,i): 
    data = pd.read_excel(file,sheet_name=sheet)   
    df = pd.DataFrame(data)
    col = df.iloc[:,i]
    return col 
#%% Read files 
f = '/Users/ulachen/OneDrive - University of Illinois - Urbana/UIUC/1_year/Atomic_Scale_Simultaions/Ni_1000.csv'

data = pd.read_csv(f, header = None, error_bad_lines=False)   
df = pd.DataFrame(data)
df = df.astype(float)
#%%Positions and Projection 
def data_C_M(f, df): 

    C=[] 
    projC=[]
    
    M = [] 
    projM=[]
    
    for i in range(len(df)): 
        if int(data_col(f,1)[i])==2: 
            xy_pos = [data_col(f,2)[i], data_col(f,3)[i]]
            xyz = [data_col(f,2)[i], data_col(f,3)[i], data_col(f,4)[i]]
            projC.append(xy_pos)
            C.append(xyz)
        elif int(data_col(f,1)[i])==1: 
            xy_pos=[data_col(f,2)[i], data_col(f,3)[i]]
            xyz = [data_col(f,2)[i], data_col(f,3)[i], data_col(f,4)[i]]
            projM.append(xy_pos)
            M.append(xyz)
            
    for i in projC: 
        for j in i: 
            float(j)
    projC = np.array(projC)
    
    for i in projM: 
        for j in i: 
            float(j)
    projM = np.array(projM) 
    
    for i in C: 
        for j in i: 
            float(j)
    C = np.array(C) 
    
    for i in M: 
        for j in i: 
            float(j)
    M = np.array(M)
    return C, M, projC, projM 

C, M, projC, projM = data_C_M(f, df)
#%%Positions and Projection Excel files 
def data_C_M(f, sh, df): 

    C=[] 
    projC=[]
    
    M = [] 
    projM=[]
    
    for i in range(len(df)): 
        if int(data_excel(f,sh,1)[i])==2: 
            xy_pos = [data_excel(f,sh,2)[i], data_excel(f,sh,3)[i]]
            xyz = [data_excel(f,sh,2)[i], data_excel(f,sh,3)[i], data_excel(f,sh,4)[i]]
            projC.append(xy_pos)
            C.append(xyz)
        elif int(data_excel(f,sh,1)[i])==1: 
            xy_pos=[data_excel(f,sh,2)[i], data_excel(f,sh,3)[i]]
            xyz = [data_excel(f,sh,2)[i], data_excel(f,sh,3)[i], data_excel(f,sh,4)[i]]
            projM.append(xy_pos)
            M.append(xyz)
            
    for i in projC: 
        for j in i: 
            float(j)
    projC = np.array(projC)
    
    for i in projM: 
        for j in i: 
            float(j)
    projM = np.array(projM) 
    
    for i in C: 
        for j in i: 
            float(j)
    C = np.array(C) 
    
    for i in M: 
        for j in i: 
            float(j)
    M = np.array(M)
    return C, M, projC, projM 
#%%
histC = np.histogram2d(projC[:,0], projC[:,1], bins=(100, 100))
histM = np.histogram2d(projM[:,0], projM[:,1], bins=(100, 100))

#%%Real and reciprocal #Metal substrate
figure(figsize=(6,6),dpi=300)
plt.scatter(projM[:,0], projM[:,1])
plt.show() 

diffM = np.absolute(np.fft.fft2(histM[0]))
plt.imshow(diffM)
plt.show()
#%%Real and reciprocal  #C
figure(figsize=(6,6),dpi=300)
plt.scatter(projC[:,0], projC[:,1])
plt.show()

diffC = np.absolute(np.fft.fft2(histC[0]))
plt.imshow(diffC)
plt.show()
#%%
f = "/Users/ulachen/Downloads/Pd_large.xlsx"
sheets = ["long", "longer"]
gr_s=[]
sk_s=[]
for sheet in range(2): 
    sh = sheets[sheet]
    data = pd.read_excel(f, sheet_name = sh)   
    df = pd.DataFrame(data)
    df = df.astype(float)
    C, M, projC, projM = data_C_M(f, sh, df)
    
    pos = [C, M]
    metal = ["Nickel", "Copper", "Palladium"]
    figure(figsize=(7,4),dpi=300)
    for atom_type in range(2): 
        N = len(pos[atom_type])
        lbox = 1
        
        displacements = displacement_table(pos[atom_type], lbox)
        distances = np.linalg.norm(displacements, axis=-1)
        dists = np.triu(distances) #lower triangular matrix to extract pair interactions 
        dists = dists[dists!=0]
        g, r = my_pair_correlation(dists, N, 100, lbox/100, lbox)
        
        if atom_type == 0:
            gr_s.append(g)
            gr_s.append("Graphene"+","+sheets[sheet])
        else: 
            gr_s.append(g)
            gr_s.append("Palladium"+","+sheets[sheet])

    maxn = 10 
    length = 1 
    kvecs = my_legal_kvecs(maxn, length)
    #Structure Factor 
    all_coor = [C, M]

    for atom_type in range(2): 
        
        sf = my_calc_sk(kvecs, all_coor[atom_type])
            
        kmags = [np.linalg.norm(kvec) for kvec in kvecs]
        unique_kmags=np.unique(kmags)
        
        unique_sk=np.zeros(len(unique_kmags))
        
        for iukmag in range(len(unique_kmags)): 
            kmag = unique_kmags[iukmag]
            idx2avg = np.where(kmags==kmag)
            unique_sk[iukmag] = np.sum(sf[idx2avg])
        
        k = (unique_kmags[1:]) #plot S(k) vs. k 
        Sk = (unique_sk[1:]) 
        
        if atom_type == 0:
            sk_s.append(Sk)
            sk_s.append("Graphene"+","+sheets[sheet])
        else: 
            sk_s.append(Sk)
            sk_s.append("Palladium"+","+sheets[sheet])
#%%
plt.plot(k,sk_s[0],'k',label=sk_s[1])
plt.plot(k,sk_s[2],'r',label=sk_s[3])
plt.plot(k,sk_s[4]+800,color="gray",label=sk_s[5])
plt.plot(k,sk_s[6]+800,'m',label=sk_s[7])
plt.xlabel('k')
plt.ylabel('S(k)')
plt.legend(loc='best')
# plt.text(2, 1150, "(a)", ha="left", va="top")
plt.tight_layout()
#%%   
plt.plot(r,gr_s[0],'k',label=gr_s[1])
plt.plot(r,gr_s[2],'r',label=gr_s[3])
plt.plot(r,gr_s[4]+40,color="gray",label=gr_s[5])
plt.plot(r,gr_s[6]+40,'m',label=gr_s[7])
plt.xlabel('r')
plt.ylabel('g(r)')
plt.legend()
# plt.text(-0.01, 39, "(b)", ha="left", va="top")
plt.tight_layout()
#%% Define files for various Temerature 
files_Temp = ["/Users/ulachen/Downloads/Ni_Temp.xlsx", "/Users/ulachen/Downloads/Cu_Temp.xlsx", "/Users/ulachen/Downloads/Pd_Temp.xlsx"]
sheets_Temp = ["1000", "1200"] 
#%%

for file in range(3): 
    for sheet in range(2): 
        f = files_Temp[file]
        sh = sheets_Temp[sheet]
        data = pd.read_excel(f, sheet_name = sh)   
        df = pd.DataFrame(data)
        df = df.astype(float)
        C, M, projC, projM = data_C_M(f, sh, df)
        
        pos = [C, M]
        tt = ["Graphene", ["Nickel", "Copper", "Palladium"]]
        figure(figsize=(7,7),dpi=300)
        for atom_type in range(2): 
            N = len(pos[atom_type])
            pair_corr = []
            step = np.arange(100)
            lbox = 1
        
            displacements = displacement_table(pos[atom_type], lbox)
            distances = np.linalg.norm(displacements, axis=-1)
            dists = np.triu(distances) #lower triangular matrix to extract pair interactions 
            dists = dists[dists!=0]
            g, r = my_pair_correlation(dists, N, 100, lbox/100, lbox)
            pair_corr.append(g)
            
            figure(file+1)
            plt.subplot(2,2,2*sheet+atom_type+1)
            plt.plot(r,g,'k')
            if atom_type == 0: 
                plt.title(tt[0]+", "+sheets_Temp[sheet]+"(K)")
            else: 
                plt.title(tt[1][file]+", "+sheets_Temp[sheet]+"(K)")
            plt.ylabel("$g(r)$")
            plt.xlabel("$r$")
            plt.tight_layout()
        maxn = 10 
        length = 1 #box length 
        kvecs = my_legal_kvecs(maxn, length)
        #Structure Factor 
        all_coor = [C, M]
    
        for atom_type in range(2): 
            
            sf = my_calc_sk(kvecs, all_coor[atom_type])
                
            kmags = [np.linalg.norm(kvec) for kvec in kvecs]
            unique_kmags=np.unique(kmags)
            
            unique_sk=np.zeros(len(unique_kmags))
            
            for iukmag in range(len(unique_kmags)): 
                kmag = unique_kmags[iukmag]
                idx2avg = np.where(kmags==kmag)
                unique_sk[iukmag] = np.sum(sf[idx2avg])
            
            k = (unique_kmags[1:]) #plot S(k) vs. k 
            Sk = (unique_sk[1:]) 
            
            figure(file+1)
            plt.subplot(2,2,2*sheet+atom_type+1)
            plt.plot(k,Sk,'k')
            if atom_type ==0: 
                plt.title(tt[0]+", "+sheets_Temp[sheet]+"(K)")
            else: 
                plt.title(tt[1][file]+", "+sheets_Temp[sheet]+"(K)")
            plt.ylabel("$S(k)$")
            plt.xlabel("$k$")
            
#%%Plots-temp
files_Temp = ["/Users/ulachen/Downloads/Ni_Temp.xlsx", "/Users/ulachen/Downloads/Cu_Temp.xlsx", "/Users/ulachen/Downloads/Pd_Temp.xlsx"]
sheets_Temp = ["1000", "1200"] 
# for file in range(3): 
sk_s = []
gr_s = []
file = 2
for sheet in range(2): 
    f = files_Temp[file]
    sh = sheets_Temp[sheet]
    data = pd.read_excel(f, sheet_name = sh)   
    df = pd.DataFrame(data)
    df = df.astype(float)
    C, M, projC, projM = data_C_M(f, sh, df)
    
    pos = [C, M]
    metal = ["Nickel", "Copper", "Palladium"]
    figure(figsize=(7,4),dpi=300)
    for atom_type in range(2): 
        N = len(pos[atom_type])
        lbox = 1
        
        displacements = displacement_table(pos[atom_type], lbox)
        distances = np.linalg.norm(displacements, axis=-1)
        dists = np.triu(distances) #lower triangular matrix to extract pair interactions 
        dists = dists[dists!=0]
        g, r = my_pair_correlation(dists, N, 100, lbox/100, lbox)
        
        if atom_type == 0:
            gr_s.append(g)
            gr_s.append("Graphene"+", T="+sheets_Temp[sheet]+"K")
        else: 
            gr_s.append(g)
            gr_s.append(metal[file]+", T="+sheets_Temp[sheet]+"K")

    maxn = 10 
    length = 1 
    kvecs = my_legal_kvecs(maxn, length)
    #Structure Factor 
    all_coor = [C, M]

    for atom_type in range(2): 
        
        sf = my_calc_sk(kvecs, all_coor[atom_type])
            
        kmags = [np.linalg.norm(kvec) for kvec in kvecs]
        unique_kmags=np.unique(kmags)
        
        unique_sk=np.zeros(len(unique_kmags))
        
        for iukmag in range(len(unique_kmags)): 
            kmag = unique_kmags[iukmag]
            idx2avg = np.where(kmags==kmag)
            unique_sk[iukmag] = np.sum(sf[idx2avg])
        
        k = (unique_kmags[1:]) #plot S(k) vs. k 
        Sk = (unique_sk[1:]) 
        
        if atom_type == 0:
            sk_s.append(Sk)
            sk_s.append("Graphene"+", T="+sheets_Temp[sheet]+"K")
        else: 
            sk_s.append(Sk)
            sk_s.append(metal[file]+", T="+sheets_Temp[sheet]+"K")
#%%
plt.plot(k,sk_s[0],'k',label=sk_s[1])
plt.plot(k,sk_s[2],'r',label=sk_s[3])
plt.plot(k,sk_s[4]+400,color="gray",label=sk_s[5])
plt.plot(k,sk_s[6]+400,'m',label=sk_s[7])
plt.xlabel('k')
plt.ylabel('S(k)')
plt.legend(loc='upper right')
plt.text(2, 1150, "(c)", ha="left", va="top")
plt.tight_layout()
#%%   
plt.plot(r,gr_s[0],'k',label=gr_s[1])
plt.plot(r,gr_s[2],'r',label=gr_s[3])
plt.plot(r,gr_s[4]+20,color="gray",label=gr_s[5])
plt.plot(r,gr_s[6]+20,'m',label=gr_s[7])
plt.xlabel('r')
plt.ylabel('g(r)')
plt.legend()
plt.text(-0.01, 39, "(c)", ha="left", va="top")
plt.tight_layout()
#%% Define files for various Orientations 
f = "/Users/ulachen/Downloads/Ni_orientation.xlsx"
sheets_orientation = ["001", "-110", "111"]

#%%Plots
colors = plt.cm.rainbow(np.linspace(0,1,3))

for sheet in range(3): 
    sh = sheets_orientation[sheet]
    data = pd.read_excel(f, sheet_name = sh)   
    df = pd.DataFrame(data)
    df = df.astype(float)
    C, M, projC, projM = data_C_M(f, sh, df)
    
    pos = C
    figure(figsize=(9,6),dpi=300)
 
    N = len(pos)
    pair_corr = []
    step = np.arange(100)
    lbox = 1

    displacements = displacement_table(pos, lbox)
    distances = np.linalg.norm(displacements, axis=-1)
    dists = np.triu(distances) #lower triangular matrix to extract pair interactions 
    dists = dists[dists!=0]
    g, r = my_pair_correlation(dists, N, 100, lbox/100, lbox)
    pair_corr.append(g)
    
    
    maxn = 10 
    length = 1 #box length 
    kvecs = my_legal_kvecs(maxn, length)
    #Structure Factor 
    all_coor = C
        
    sf = my_calc_sk(kvecs, all_coor)
        
    kmags = [np.linalg.norm(kvec) for kvec in kvecs]
    unique_kmags=np.unique(kmags)
    
    unique_sk=np.zeros(len(unique_kmags))
    
    for iukmag in range(len(unique_kmags)): 
        kmag = unique_kmags[iukmag]
        idx2avg = np.where(kmags==kmag)
        unique_sk[iukmag] = np.sum(sf[idx2avg])
    
    k = (unique_kmags[1:]) #plot S(k) vs. k 
    Sk = (unique_sk[1:]) 
    
    figure(1)
    plt.subplot(3,3,sheet+1)
    plt.plot(r,g,'k') 
    plt.title("Graphene, along Nickel["+sheets_orientation[sheet]+"]")
    plt.ylabel("$g(r)$")
    plt.xlabel("$r$")
    plt.tight_layout()    
    
    plt.subplot(3,3,sheet+4)
    plt.plot(k,Sk,'k')
    plt.title("Graphene, along Nickel["+sheets_orientation[sheet]+"]")
    plt.ylabel("$S(k)$")
    plt.xlabel("$k$")
    plt.tight_layout()       
    
#%%Plots Nickel diff direction 

for sheet in range(3): 
    sh = sheets_orientation[sheet]
    data = pd.read_excel(f, sheet_name = sh)   
    df = pd.DataFrame(data)
    df = df.astype(float)
    C, M, projC, projM = data_C_M(f, sh, df)
    
    pos = M
    figure(figsize=(9,6),dpi=300)
 
    N = len(pos)
    pair_corr = []
    step = np.arange(100)
    lbox = 1

    displacements = displacement_table(pos, lbox)
    distances = np.linalg.norm(displacements, axis=-1)
    dists = np.triu(distances) #lower triangular matrix to extract pair interactions 
    dists = dists[dists!=0]
    g, r = my_pair_correlation(dists, N, 100, lbox/100, lbox)
    pair_corr.append(g)
    
    
    maxn = 10 
    length = 1 #box length 
    kvecs = my_legal_kvecs(maxn, length)
    #Structure Factor 
    all_coor = M
        
    sf = my_calc_sk(kvecs, all_coor)
        
    kmags = [np.linalg.norm(kvec) for kvec in kvecs]
    unique_kmags=np.unique(kmags)
    
    unique_sk=np.zeros(len(unique_kmags))
    
    for iukmag in range(len(unique_kmags)): 
        kmag = unique_kmags[iukmag]
        idx2avg = np.where(kmags==kmag)
        unique_sk[iukmag] = np.sum(sf[idx2avg])
    
    k = (unique_kmags[1:]) #plot S(k) vs. k 
    Sk = (unique_sk[1:]) 
    
    figure(1)
    plt.subplot(3,3,sheet+1)
    plt.plot(r,g,'k') 
    plt.title("Nickel, along"+sheets_orientation[sheet])
    plt.ylabel("$g(r)$")
    plt.xlabel("$r$")
    plt.tight_layout()    
    
    plt.subplot(3,3,sheet+4)
    plt.plot(k,Sk,'k')
    plt.title("Nickel, along"+sheets_orientation[sheet])
    plt.ylabel("$S(k)$")
    plt.xlabel("$k$")
    plt.tight_layout()   
#%% Pair Correlation function 


files = ["/Users/ulachen/Downloads/Ni_1000.csv", "/Users/ulachen/Downloads/Cu_1000.csv", "/Users/ulachen/Downloads/Pd_1000.csv"]

for file in range(3): 
    f = files[file]
    data = pd.read_csv(f, header = None, error_bad_lines=False)   
    df = pd.DataFrame(data)
    df = df.astype(float)
    C, M, projC, projM = data_C_M(f, df)
    
    pos = [C, M]
    tt = ["Graphene", ["Nickel", "Copper", "Palladium"]]
    figure(figsize=(7,3),dpi=300)
    for atom_type in range(len(pos)): 
        N = len(pos[atom_type])
        pair_corr = []
        step = np.arange(100)
        lbox = 1
    
        displacements = displacement_table(pos[atom_type], lbox)
        distances = np.linalg.norm(displacements, axis=-1)
        dists = np.triu(distances) #lower triangular matrix to extract pair interactions 
        dists = dists[dists!=0]
        g, r = my_pair_correlation(dists, N, 100, lbox/100, lbox)
        pair_corr.append(g)
        
        
        plt.subplot(1,2,atom_type+1)
        plt.plot(r,g,'k')
        if atom_type == 0: 
            plt.title(tt[0])
        else: 
            plt.title(tt[1][file])
        plt.ylabel("$g(r)$")
        plt.xlabel("$r$")
    maxn = 10 
    length = 1 #box length 
    kvecs = my_legal_kvecs(maxn, length)
    
    figure(figsize=(7,3),dpi=300)
    #Structure Factor 
    all_coor = [C, M]

    for atom_type in range(len(all_coor)): 
        
        sf = my_calc_sk(kvecs, all_coor[atom_type])
            
        kmags = [np.linalg.norm(kvec) for kvec in kvecs]
        unique_kmags=np.unique(kmags)
        
        unique_sk=np.zeros(len(unique_kmags))
        
        for iukmag in range(len(unique_kmags)): 
            kmag = unique_kmags[iukmag]
            idx2avg = np.where(kmags==kmag)
            unique_sk[iukmag] = np.sum(sf[idx2avg])
        
        k = (unique_kmags[1:]) #plot S(k) vs. k 
        Sk = (unique_sk[1:]) 
        
        plt.subplot(1,2,atom_type+1)
        plt.plot(k,Sk,'k')
        if atom_type ==0: 
            plt.title(tt[0])
        else: 
            plt.title(tt[1][file])
        plt.ylabel("$S(k)$")
        plt.xlabel("$k$")


        