import os
import sys
import numpy as np
from math import pi,sqrt,e


def k0(f):
    return f*2*pi/3*pow(10,-8)

def solve(matrix):
    for i in range(len(matrix) - 1):
        for j in range(len(matrix[0]) - 1):
            if abs(matrix[i][j] - matrix[i + 1][j + 1])>pow(10,-7):
                return False
    return True

class ProgressBar:
    def __init__(self, total, length=50, title='building G'):
        self.total = total
        self.length = length
        self.title = title

    def update(self, progress):
        percent = "{0:.1f}".format(100 * (progress / float(self.total)))
        filled_length = int(self.length * progress // self.total)+1

        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'

        filled = GREEN + '-' * filled_length + RESET
        unfilled = RED + '-' * (self.length - filled_length) + RESET
        bar = filled + unfilled

        sys.stdout.write(f'\r{self.title} {bar} {percent}% (￣o￣) . z Z')
        if percent == "100.0":
            sys.stdout.write(f'\r{self.title} {bar} {percent}% ╰(*°▽°*)╯')
        sys.stdout.flush()

class GBuilder:
    def __init__(self, xstep, ystep, xpoints, ypoints, h):
        self.xstep = xstep
        self.ystep = ystep
        self.xpoints = int(xpoints)
        self.ypoints = int(ypoints)
        self.h = h
        self.m = np.full((self.xpoints,self.ypoints),-1,dtype=complex)

        z = sqrt(float(3/5))/0.5
        self.gauss = [[-z,z,z,-z,0,z,0,-z,0],
                      [-z,-z,z,z,-z,0,z,0,0],
                      [25/324,25/324,25/324,25/324,40/324,40/324,40/324,40/324,64/324]]

    def decode(self,num):
        xindex = num%self.xpoints
        yindex = num//self.ypoints
        return xindex,yindex

    # 矩量法
    def mom(self, num, freq):
        g=[]

        xindex_,yindex_=self.decode(num)
        for i in range(0,self.xpoints*self.ypoints):
            xindex,yindex=self.decode(i)
            xdiff = abs(xindex-xindex_)
            ydiff = abs(yindex-yindex_)
            gblock = 0
            if self.m[xdiff,ydiff] == -1 and self.m[ydiff,xdiff] == -1:
                #高斯数值积分
                for j in range(0,9):
                    r = sqrt(pow((xindex_-xindex+self.gauss[0][j])*self.xstep,2)+pow((yindex_-yindex+self.gauss[1][j])*self.ystep,2)+pow(self.h,2))
                    gblock += pow(e,-1*k0(freq)*r*1j)/(4*pi*pow(r,2))*(k0(freq)*1j+1/r)*self.h*-1*self.gauss[2][j]
                self.m[xdiff,ydiff] = gblock
            else:
                gblock = self.m[xdiff,ydiff] if self.m[xdiff,ydiff]!=-1 else self.m[ydiff,xdiff]
            g.append(gblock)

        return g

    def build(self, freq):
        G=[]
        if os.path.isfile("G.npy"):
            G = np.load("G.npy")
        else:
            progress = ProgressBar(self.xpoints*self.ypoints)
            for i in range(0,self.xpoints*self.ypoints):
                G.append(self.mom(i, freq))
                progress.update(i)
            np.save("G.npy", G)

        return G

# 共轭梯度法求解,E=GM
# https://blog.csdn.net/weixin_45933967/article/details/145704404
def PCG(E,G):
    cnt = 0
    E = np.array(E)
    G = np.array(G)
    x = np.full(G.shape[1],0,dtype=complex)
    r = E-np.dot(G,x)
    rTr = np.vdot(r, r)
    p = E-np.dot(G,x)
    while rTr.real > 0.0001 and cnt <21:
        alpha = rTr/np.vdot(p,np.dot(G,p))
        x = x +alpha*p
        r = E-np.dot(G,x)
        beta = -1*np.vdot(r, r)/rTr
        p = r+beta*p
        rTr = np.vdot(r, r)
        cnt+=1
        print(f"{cnt} rTr={rTr}")
    return x


if __name__ == "__main__":
    E=[1,2]
    G=[[1,2],[1,4]]
    x = PCG(E, G)
    print(x)


