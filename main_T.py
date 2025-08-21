import load
from rebuild import Rebuilder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Edataloader = load.DataLoader("E_origin.fld")
    Hdataloader = load.DataLoader("H_origin.fld")
    E,h,xpoints,xstep,ypoints,ystep = Edataloader.get_data()
    H,_,_,_,_,_ = Hdataloader.get_data()
    rebuilder = Rebuilder(E[:1458],H[:1458],h,xstep,ystep,xpoints,ypoints,z0=480,dxlenth=0.104,dylenth=0.104,dxpoints=26,dypoints=26,f=5e9)
    dipole = rebuilder.solve(state="T")
    Edataloader = load.DataLoader("E_30mm.fld")
    Hdataloader = load.DataLoader("H_30mm.fld")
    E, h, xpoints, xstep, ypoints, ystep = Edataloader.get_data()
    H, _, _, _, _, _ = Hdataloader.get_data()
    farrebuilder = Rebuilder(E[:1458], H[:1458], h, xstep, ystep, xpoints, ypoints, dxlenth=0.104, dylenth=0.104, dxpoints=26,dypoints=26, f=5e9)
    farrebuilder.T_build()
    H_diople = farrebuilder.T @ dipole
    plt.figure()
    sns.heatmap(np.angle(H_diople[729*2:729*2+729]).reshape((27, 27)), cmap="YlOrRd")
    plt.title("Custom Labeled Heatmap")
    plt.show()
    plt.figure()
    sns.heatmap(np.angle(H[:729]).reshape((27, 27)), cmap="YlOrRd")
    plt.title("Custom Labeled Heatmap")
    plt.show()
    print(f"Hx={np.linalg.norm(H_diople[729*2:729*2+729] - H[:729])}")
    plt.figure()
    sns.heatmap(np.angle(H_diople[729*2+729:729*2+729*2]).reshape((27, 27)), cmap="YlOrRd")
    plt.title("Custom Labeled Heatmap")
    plt.show()
    plt.figure()
    sns.heatmap(np.angle(H[729:1458]).reshape((27, 27)), cmap="YlOrRd")
    plt.title("Custom Labeled Heatmap")
    plt.show()
    print(f"Hy={np.linalg.norm(H_diople[729*2+729:729*2+729*2] - H[729:1458])}")







