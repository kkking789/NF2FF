import os
import sys
import numpy as np
from math import pi,sqrt,e

from matplotlib import pyplot as plt

import load


def k0(f):
    c=3*pow(10,8)
    return f*2*pi/c

class ProgressBar:
    def __init__(self, total, length=50, title='building'):
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
                      # [0, 0, 0, 0, 0, 0, 0, 0, 1]]

    def decode(self,num):
        xindex = num%self.xpoints
        yindex = num//self.xpoints
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

        return np.array(G)

# 共轭梯度法求解,E=GM,没用,这个只能解正定方程
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

def diff(x,y):
    diffsum=0
    sum=0
    for cnt in range(len(x)):
        diffsum+=(abs(x[cnt]-y[cnt]))**2
        sum+=abs(y[cnt])**2
    return sqrt(diffsum/sum)


class SourceRebuilder:

    def __init__(self, E, H, h, f, xstep, ystep, xpoints, ypoints, hd=0.001):
        self.E = E
        self.H = H
        self.h = h
        self.f = f
        self.X = 0
        self.hd = hd
        self.xpoints = xpoints
        self.ypoints = ypoints
        self.xstep = xstep
        self.ystep = ystep
        self.xstep2 = xstep**2
        self.ystep2 = ystep**2
        self.k0 = k0(f)
        self.gammaE = -1j*self.k0*120/4
        self.gammaH = self.k0/(4*pi)
        self.Emax = np.max(np.abs(E))
        self.Hmax = np.max(np.abs(H))
        self.bias = [np.abs(bias) for bias in np.append(E,H)]

    def r1(self, x0, x1, y0, y1):
        return sqrt((x0-x1)**2*self.xstep2 + (y0-y1)**2*self.ystep2+self.h**2)
    def r2(self, x0, x1, y0, y1):
        return sqrt((x0-x1)**2*self.xstep2 + (y0-y1)**2*self.ystep2+(self.h+2*self.hd)**2)
    def q(self,r):
        if r > pow(10,5) or r ==-1:#远场
            return -1,0,-1j
        fr = e**(-1j*self.k0*r)/r
        q1 = (3/(self.k0*r)**2 + 3j/(self.k0*r)-1)*fr
        q2 = (2/(self.k0*r)**2 + 2j/(self.k0*r))*fr
        q3 = (1/(self.k0*r)+1j)*fr
        return q1,q2,q3

    def Ts(self, x0, x1, y0, y1):
        z = self.h+self.hd
        hd = self.hd
        gammaE = self.gammaE
        gammaH = self.gammaH
        k0 = self.k0
        r1 = self.r1(x0, x1, y0, y1)
        r2 = self.r2(x0, x1, y0, y1)
        q11,q12,q13 = self.q(r1)
        q21,q22,q23 = self.q(r2)

        x0 = x0*self.xstep
        x1 = x1*self.xstep
        y0 = y0*self.ystep
        y1 = y1*self.ystep

        Texpz = gammaE*(
            (z-hd)*(x0-x1)/r1**2*q11+
            (z+hd)*(x0-x1)/r2**2*q21
        )
        Texmx = 0
        Texmy = gammaE*k0*(
            (z-hd)/r1*q13+
            (z+hd)/r2*q23
        )
        Teypz = gammaE*(
            (y0-y1)*(z-hd)/r1**2*q11+
            (y0-y1)*(z+hd)/r2**2*q21
        )
        Teymx = gammaE*k0*(
            -1*(z-hd)/r1*q13+
            -1*(z+hd)/r2*q23
        )
        Teymy = 0
        Thxpz = gammaH*(
            -1*(y0-y1)/r1*q13+
            -1*(y0-y1)/r2*q23
        )
        Thxmx = gammaH*k0*(
            -1*((y0-y1)**2+(z-hd)**2)/r1**2*q11+
            q12+
            -1*((y0-y1)**2+(z+hd)**2)/r2**2*q21+
            q22
        )
        Thxmy = gammaH*k0*(
            (x0-x1)*(y0-y1)/r1**2*q11+
            (x0-x1)*(y0-y1)/r2**2*q21
        )
        Thypz = gammaH*(
            (x0-x1)/r1*q13+
            (x0-x1)/r2*q23
        )
        Thymx = gammaH*k0*(
            (x0-x1)*(y0-y1)/r1**2*q11+
            (x0-x1)*(y0-y1)/r2**2*q21
        )
        Thymy = gammaH*k0*(
            -1*((z-hd)**2+(x0-x1)**2)/r1**2*q11+
            q12+
            -1*((z+hd)**2+(x0-x1)**2)/r2**2*q21+
            q22
        )

        return [Texpz,Texmx/k0,Texmy/k0,Teypz,Teymx/k0,Teymy/k0,\
                Thxpz,Thxmx/k0,Thxmy/k0,Thypz,Thymx/k0,Thymy/k0]

    def T(self):
        size=self.xpoints*self.ypoints
        Tarray=np.zeros((size*4,size*3),dtype=complex)
        bar = ProgressBar(size**2)
        for x0 in range(self.xpoints):
            for y0 in range(self.ypoints):
                for x1 in range(60):
                    for y1 in range(60):
                        cnt = x0*size*self.ypoints+y0*size+x1*self.ypoints+y1
                        mid = self.Ts(x0,x1,y0,y1)
                        Tarray[x1*self.ypoints+y1,x0*self.ypoints+y0] = mid[0]
                        Tarray[x1*self.ypoints+y1,x0*self.ypoints+y0+size] = mid[1]
                        Tarray[x1*self.ypoints+y1,x0*self.ypoints+y0+size*2] = mid[2]
                        Tarray[x1 * self.ypoints + y1+size, x0 * self.ypoints + y0] = mid[3]
                        Tarray[x1 * self.ypoints + y1 + size, x0 * self.ypoints + y0+size] = mid[4]
                        Tarray[x1 * self.ypoints + y1 + size, x0 * self.ypoints + y0+size] = mid[5]
                        Tarray[x1 * self.ypoints + y1 + size*2, x0 * self.ypoints + y0] = mid[6]
                        Tarray[x1 * self.ypoints + y1 + size * 2, x0 * self.ypoints + y0+size] = mid[7]
                        Tarray[x1 * self.ypoints + y1 + size * 2, x0 * self.ypoints + y0+size*2] = mid[8]
                        Tarray[x1 * self.ypoints + y1 + size * 3, x0 * self.ypoints + y0] = mid[9]
                        Tarray[x1 * self.ypoints + y1 + size * 3, x0 * self.ypoints + y0 + size] = mid[10]
                        Tarray[x1 * self.ypoints + y1 + size * 3, x0 * self.ypoints + y0 + size * 2] = mid[11]
                        bar.update(cnt)

        return Tarray

    def solve(self, E, H, b=0.0001, save=True, limits=(5,1e-9),itera=True):
        if save:
            if os.path.isfile("T.npy"):
                T = np.load("T.npy")
                T = np.matrix(T)
            else:
                T = np.matrix(self.T())
                T = np.matrix([T[:][i] / self.bias[i] for i in range(len(T[:]))])
                np.save("T.npy", T)
            if os.path.isfile("VH.npy") and os.path.isfile("S.npy") and os.path.isfile("U.npy"):
                VH = np.matrix(np.load("VH.npy"))
                U = np.matrix(np.load("U.npy"))
                S = np.load("S.npy")
            else:
                U, S, VH = np.linalg.svd(T)
                VH = np.matrix(VH)
                np.save("S.npy", S)
                np.save("VH.npy", VH)
                np.save("U.npy", U)
            if os.path.isfile("F.npy"):
                F = np.load("F.npy")
            else:
                Ex = np.reshape([e[0] / abs(e[0]) for e in E], (1, -1))
                Ey = np.reshape([e[1] / abs(e[1]) for e in E], (1, -1))
                Hx = np.reshape([h[0] / abs(h[0]) for h in H], (1, -1))
                Hy = np.reshape([h[1] / abs(h[1]) for h in H], (1, -1))
                E = np.append(Ex, Ey)
                H = np.append(Hx, Hy)
                F = np.reshape([E, H], (1, -1)).T
                np.save("F.npy", F)
        else:
            T = np.matrix(self.T())
            T = np.matrix([T[:][i] / self.bias[i] for i in range(len(T[:]))])
            U, S, VH = np.linalg.svd(T)
            VH = np.matrix(VH)
            Ex = np.reshape([e[0] / abs(e[0]) for e in E], (1, -1))
            Ey = np.reshape([e[1] / abs(e[1]) for e in E], (1, -1))
            Hx = np.reshape([h[0] / abs(h[0]) for h in H], (1, -1))
            Hy = np.reshape([h[1] / abs(h[1]) for h in H], (1, -1))
            E = np.append(Ex, Ey)
            H = np.append(Hx, Hy)
            F = np.reshape([E, H], (1, -1)).T
            np.save("T.npy", T)
            np.save("S.npy", S)
            np.save("VH.npy", VH)
            np.save("U.npy", U)
            np.save("F.npy", F)

        TH = T.H
        V = np.array(VH.H)

        if itera:
            scope=[limits[0],(limits[0]+limits[1])/2,limits[1]]
            tol=[-1]*3
            mintor=1e5
            midtor=1e5
            cnt=0
            while True:
                for i in range(3):
                    if tol[i]==-1:
                        ambda=scope[i]*S[0]
                        S_ = np.diag([1 / (xi ** 2 + ambda ** 2) for xi in S])
                        X = V @ S_ @ VH @ TH @ F
                        Fs = T @ X
                        sigmaEx = diff(Fs[0:2601], F[0:2601])
                        sigmaEy = diff(Fs[2601:5202], F[2601:5202])
                        sigmaHx = diff(Fs[5202:7803], F[5202:7803])
                        sigmaHy = diff(Fs[7803:10404], F[7803:10404])
                        tol[i]=sigmaEx+sigmaEy+sigmaHx+sigmaHy
                        print(f"for {scope[i]} tol: {tol[i]}\n"
                              f"Ex:{sigmaEx},Ey:{sigmaEy},Hx:{sigmaHx},Hy:{sigmaHy}")
                    if tol[i]<mintor:
                        midtor = mintor
                        mintor = tol[i]
                    elif mintor<tol[i]<midtor:
                        midtor = tol[i]
                if midtor!=tol[i] and mintor!=tol[i]:
                    midtor = tol[i]
                newmax = scope[tol.index(midtor)]
                newmin = scope[tol.index(mintor)]
                scope = [newmax, (newmin + newmax) / 2, newmin]
                tol = [midtor, -1, mintor]
                cnt += 1
                if (midtor-mintor)<0.05:
                    print(f"can't find best solution\nepoches:{cnt},scope:{scope},tol:{tol}")
                    break
                print(f"epoches:{cnt},scope:{scope},tol:{tol}")
                if mintor<0.1:
                    print(f"find best solution b={scope[2]} in {cnt} epoch,tolerance is {tol[2]}\n"
                          f"Ex:{sigmaEx},Ey:{sigmaEy},Hx:{sigmaHx},Hy:{sigmaHy}")
                    break
        else:
            ambda = b*S[0]
            S = np.diag([1/(xi**2+ambda**2) for xi in S])
            X = V@S@VH@TH@F
            Fs = T@X
            sigmaEx = diff(Fs[0:2601],F[0:2601])
            sigmaEy = diff(Fs[2601:5202], F[2601:5202])
            sigmaHx = diff(Fs[5202:7803], F[5202:7803])
            sigmaHy = diff(Fs[7803:10404], F[7803:10404])
            print(sigmaEx,sigmaEy,sigmaHx,sigmaHy)

        self.X=X
        return X


    # def far(self):

        
        





if __name__ == "__main__":
    print(e)


