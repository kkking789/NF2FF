import load
from rebuild import Rebuilder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Edataloader = load.DataLoader("nf_E.fld")
    Hdataloader = load.DataLoader("nf_H.fld")
    E,h,xpoints,xstep,ypoints,ystep = Edataloader.get_data()
    H,_,_,_,_,_ = Hdataloader.get_data()
    Hlist=[]
    for i in range(3):
        for Hdata in H:
            Hlist.append(Hdata[i])
    H=np.array(Hlist)
    rebuilder = Rebuilder(E,H,h,xstep,ystep,xpoints,ypoints,dxpoints=40,dypoints=40,f=2.5e9)
    cost,dipole = rebuilder.solve()





