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
    rebuilder = Rebuilder(E,H,h,xstep,ystep,xpoints,ypoints,dxlenth=0.104,dylenth=0.104,dxpoints=26,dypoints=26,f=5e9)
    dipole = rebuilder.solve(state="Q")
    Edataloader = load.DataLoader("E_30mm.fld")
    Hdataloader = load.DataLoader("H_30mm.fld")
    E, h, xpoints, xstep, ypoints, ystep = Edataloader.get_data()
    H, _, _, _, _, _ = Hdataloader.get_data()
    farrebuilder = Rebuilder(E, H, h, xstep, ystep, xpoints, ypoints, dxlenth=0.104, dylenth=0.104, dxpoints=26,dypoints=26, f=5e9)
    farrebuilder.Q_build()
    H_diople = farrebuilder.Q @ dipole
    plt.figure()
    sns.heatmap(np.abs(H_diople[729*3:729*4]).reshape((27, 27)), cmap="YlOrRd")
    plt.title("Custom Labeled Heatmap")
    plt.show()
    plt.figure()
    sns.heatmap(np.abs(H[:729]).reshape((27, 27)), cmap="YlOrRd")
    plt.title("Custom Labeled Heatmap")
    plt.show()
    print(f"Hx={np.linalg.norm(H_diople[729*3:729*4] - H[:729])}")
    plt.figure()
    sns.heatmap(np.abs(H_diople[729*4:729*5]).reshape((27, 27)), cmap="YlOrRd")
    plt.title("Custom Labeled Heatmap")
    plt.show()
    plt.figure()
    sns.heatmap(np.abs(H[729:1458]).reshape((27, 27)), cmap="YlOrRd")
    plt.title("Custom Labeled Heatmap")
    plt.show()
    print(f"Hy={np.linalg.norm(H_diople[729*4:729*5] - H[729:1458])}")
    plt.figure()
    sns.heatmap(np.abs(H_diople[729*5:729*6]).reshape((27, 27)), cmap="YlOrRd")
    plt.title("Custom Labeled Heatmap")
    plt.show()
    plt.figure()
    sns.heatmap(np.abs(H[1458:2187]).reshape((27, 27)), cmap="YlOrRd")
    plt.title("Custom Labeled Heatmap")
    plt.show()
    print(f"Hz={np.linalg.norm(H_diople[729*5:729*6] - H[1458:2187])}")






