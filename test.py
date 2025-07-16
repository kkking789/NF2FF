from math import pi,sqrt,e

z = sqrt(float(3 / 5)) / 0.5
gauss = [[-z, z, z, -z, 0, z, 0, -z, 0],
              [-z, -z, z, z, -z, 0, z, 0, 0],
              [25 / 324, 25 / 324, 25 / 324, 25 / 324, 40 / 324, 40 / 324, 40 / 324, 40 / 324, 64 / 324]]
xindex_,yindex_=50,50
xindex,yindex=0,40
h=6
gblock = 0
freq = 75000000
def k0(f):
    return f*2*pi/3*pow(10,-8)
for j in range(0, 9):
    r = sqrt(pow((xindex_ - xindex + gauss[0][j]), 2) + pow(
        (yindex_ - yindex + gauss[1][j]), 2) + pow(h, 2))
    gblock += pow(e, -1 * k0(freq) * r * 1j) / (4 * pi * pow(r, 2)) * (k0(freq) * 1j + 1 / r) * h * -1 * \
              gauss[2][j]
print(gblock)