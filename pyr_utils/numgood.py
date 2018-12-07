import numpy as np
import math
def numgood(N,M,B):
    z=totnum(N,M)
    #mat=np.zeros((N,M),dtype=np.int64)
    mat=[[0 for i in range(M)] for j in range(N)]
    mat[0][0]=1
    reachable=np.zeros((N,M))
    reachable[0,0]=1
    for x in range(M-1):
        for y in range(N):
            if (reachable[y,x]):
                balls_to_place=N-y
                max_for_later=B*(M-x)
                min_for_now=np.max((balls_to_place-max_for_later,0))
                max_for_now=np.min((balls_to_place,B))
                for b in range(min_for_now,max_for_now):
                    mat[y+b][x+1]+=mat[y][x]
                    reachable[y+b,x+1]=1
    return 1-np.sum([m[-1] for m in mat])/z


def totnum(stars,bars):
    return choose(stars+bars-1,bars-1)

def choose(n,k):
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
