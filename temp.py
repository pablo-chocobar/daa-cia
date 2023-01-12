a = "acgtcaacacac"
b = "acactcacacac"

n = len(a)
m = len(b)

import numpy as np

mat = np.zeros((n+1,m+1), dtype = int)
par = np.zeros((n+1,m+1,2), dtype = int)

match = 2
mism = -1
gap = -1
ex = 0
#for i in range(0,n+1):
 #   mat[0,i] = ex + i*gap 
  #  mat[i, 0] = ex + i*gap
        
for i in range(1,n+1):
    for j in range(1,m+1):
        if(a[i-1] == b[j-1]):
            mat[i,j] = max(mat[i-1, j-1] + match,0)
            par[i,j] = [i-1, j -1]
        else:
            x = mat[i-1, j] + gap + ex
            y = mat[i, j-1] + gap + ex
            z = mat[i-1, j-1] + mism
        
            mat[i,j] = max(x,y,z,0)
            t = max(x,y,z)
            if(z == t):
                par[i,j] = [i-1,j-1]
            if(x == t):
                par[i,j] = [i-1, j]
            if(y == t):
                par[i,j] = [i, j-1]
print(mat)
print(par[:,:,0], par[:,:,1])
trm = [[n,m]]
def trace(par, trm, x, y):
    if x == 0 and y == 0:
        return trm
    if x==1 and y==1:
        return trm
    else:
        trm.append(par[x,y])
        return trace(par, trm, int(par[x,y][0]), int(par[x,y][1]))
  
trace(par, trm, len(a), len(b))
print(trm)

string1 = ""
string2=""

for i in range(0,len(trm)-1):
    temp = trm[i]
    if(b[int(temp[1]-1)]==a[int(temp[0]-1)]):
        string1 = b[int(temp[1]-1)]+string1
        string2=a[int(temp[0]-1)]+string2

    elif( b[int(temp[1]-1)] != a[int(temp[0]-1)]) :
        if(trm[i+1][1]==temp[1]-1):
            string1 = '_'+string1
            string2 = a[int(temp[1]-1)]+ string2 
        else:
            string1 =b[int(temp[0]-1)] + string1
            string2 = '_' +string2 

if(b[0]==a[0]):
    string1 = a[0] + string1  
    string2 = b[0] + string2
else:
    string1 ='_' + string1
    string2 ='_' + string2

print(string1)
print(string2)