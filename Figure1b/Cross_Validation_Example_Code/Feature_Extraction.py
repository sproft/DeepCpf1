from numpy import *
from Bio.SeqUtils import MeltingTemp as mt
import RNA
    
def Feature_Extraction(lines):
    data_n = len(lines)
    DATA_X = zeros((data_n,34,4), dtype=int)
    DATA_Y = zeros((data_n,), dtype=float)
    SEQ = []
    
    for l in range(data_n):
        data = lines[l].split()
        seq = data[1]
        SEQ.append(seq)
        for i in range(34):
            if seq[i] in "Aa":   DATA_X[l, i, 0] = 1
            elif seq[i] in "Cc": DATA_X[l, i, 1] = 1
            elif seq[i] in "Gg": DATA_X[l, i, 2] = 1
            elif seq[i] in "Tt": DATA_X[l, i, 3] = 1
        DATA_Y[l] = float(data[0])
    
    #Feature Extraction
    DATA_X_FE = zeros((data_n,689), dtype=float)
    for l in range(data_n):
        #position-independent nucleotides and dinucleotides (4 + 4^2 = 20)
        for i in range(4):
            DATA_X_FE[l,i] = sum(DATA_X[l,:,i])
        for i in range(4,20):
            DATA_X_FE[l,i] = Dinucleotide_FE(DATA_X[l], (i-4)/4, (i-4)%4)
            
        #position-dependent nucleotides and dinucleotides ( 4*34 + (4^2 * 33) = 664)
        for i in range(20, 156):
            DATA_X_FE[l,i] = DATA_X[l, (i-20)/4, (i-20)%4]
        for i in range(156,684):
            DATA_X_FE[l,i] = Dinucleotide_FE(DATA_X[l, (i-156)/16:(i-156)/16+2, :], ((i-156)%16)/4, ((i-156)%16)%4)
            
        #Melting temperatiure (1)
        DATA_X_FE[l,684] = mt.Tm_NN(SEQ[l])
           
        #GC count (3)
        DATA_X_FE[l,685] = SEQ[l].count("G") + SEQ[l].count("g") + SEQ[l].count("C") + SEQ[l].count("c")
        if DATA_X_FE[l,685] <= 9:
            DATA_X_FE[l,686] = 1
            DATA_X_FE[l,687] = 0
        else:
            DATA_X_FE[l,686] = 0
            DATA_X_FE[l,687] = 1
            
        #Free energy
        DATA_X_FE[l,688] = RNA.fold(SEQ[l])[1]

    return DATA_X_FE, DATA_Y

def Dinucleotide_FE(X, a, b):
    num = 0
    for i in range(X.shape[0]-1):
        if X[i,a] == 1 and X[i+1,b] == 1:
            num += 1
    return num