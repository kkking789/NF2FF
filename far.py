import numpy as np
from math import cos, sin, pi


class Far:
    def __init__(self, X, rebuilder, num=360, r=300):
        self.X=X
        self.k = rebuilder.k
        self.K_h = rebuilder.K_h
        self.K_e = rebuilder.K_e
        self.rebuilder=rebuilder
        self.measure = rebuilder.measure
        self.rebuild = rebuilder.rebuild
        self.num = num
        self.r = r


    def farbuilder(self):
        gamma_E = self.K_e
        gamma_H = self.K_h
        k = self.k
        r=self.r
        X = self.X
        anglestep = 2*pi/self.num
        measure = self.measure  # 测量点网格
        rebuild = self.rebuild  # 重建点网格

        for M in range(self.num):
            Ex=0
            Ey=0
            Ez=0
            Hx=0
            Hy=0
            Hz=0
            angle = anglestep*M
            for m in range(rebuild.total):
                Pz = X[m]
                Mx = X[m+rebuild.total]
                My = X[m+rebuild.total*2]
                Ex += (sin(angle)*cos(angle)*-1*2*Pz+sin(angle)*1j*2*k*My)*gamma_E
                Ey += (-1*sin(angle)*2*1j*k*Mx)*gamma_E
                Ez += (cos(angle)**2*2*Pz+-1*cos(angle)*1j*2*My)*gamma_E









