from numpy import linalg as LA

def condi(M):
    return LA.cond(M)