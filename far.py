import numpy as np
from math import e
from rebuild import diff


class Far:
    def __init__(self, X, rebuilder):
        self.X=X
        self.xpoints=rebuilder.solvex
        self.ypoints=rebuilder.solvey
        self.rebuilder=rebuilder
        self.T = []

    def near(self,H,E,h):
        self.rebuilder.h=h
        self.T=self.rebuilder.T(normalize=False)
        np.save(f"T{self.rebuilder.h}",self.T)
        Ex = np.reshape([e[0] for e in E], (1, -1))
        Ey = np.reshape([e[1] for e in E], (1, -1))
        Hx = np.reshape([h[0] for h in H], (1, -1))
        Hy = np.reshape([h[1] for h in H], (1, -1))
        E = np.append(Ex, Ey)
        H = np.append(Hx, Hy)
        F = np.append(E, H)
        Fs = np.array(self.T) @ np.array(self.X)
        sigmaEx = diff(Fs[0:2601], F[0:2601])
        sigmaEy = diff(Fs[2601:5202], F[2601:5202])
        sigmaHx = diff(Fs[5202:7803], F[5202:7803])
        sigmaHy = diff(Fs[7803:10404], F[7803:10404])
        tol=diff(Fs,F)
        print(f"tol: {tol},Ex:{sigmaEx},Ey:{sigmaEy},Hx:{sigmaHx},Hy:{sigmaHy}")
        np.save(f"E_{h}",abs(E))
        np.save(f"E_rebuild",abs(Fs))





