import numpy as np
from math import e
from rebuild import diff


class Far:
    def __init__(self, X, rebuilder,xpoints=30, ypoints=30):
        self.X=X
        self.xpoints=xpoints
        self.ypoints=ypoints
        self.rebuilder=rebuilder
        self.T = []

    def near(self,h,path):
        self.rebuilder.h=h
        self.T=self.rebuilder.T()
        datas=[]
        with open(path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line[0] == "G" or line[0] == "X":
                    continue
                datas.append([float(mid) for mid in line.split()])
            comE = [(data[3] * e ** (data[4] * 1j), data[5] * e ** (data[6] * 1j)) for data in datas]
            Ex = np.reshape([e[0] for e in comE], (1, -1))
            Ey = np.reshape([e[1] for e in comE], (1, -1))
            E = np.append(Ex, Ey)
        Er = self.T @ self.X
        np.save(f"E_{h}",abs(E))
        np.save(f"E_rebuild",abs(Er))
        tor = diff(abs(E),abs(Er))
        print(f"tor = {tor}")




