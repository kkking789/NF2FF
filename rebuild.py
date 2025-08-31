import os.path
import sys
import math
from math import e
import numpy as np
import pyswarms as ps
from sko.DE import DE
import random
from sko.SA import SA
from sko.GA import GA


class ProgressBar:
	def __init__(self, total, length=50, title='building'):
		self.total = total
		self.length = length
		self.title = title
		self.progress = 0

	def update(self):
		progress = self.progress
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
		self.progress+=1

class Grid:
	def __init__(self,xstep,ystep,xpoints,ypoints):
		self.xstep = xstep
		self.ystep = ystep
		self.xpoints = xpoints
		self.ypoints = ypoints
		self.total = xpoints*ypoints

class Rebuilder:
	def __init__(self,E,H,h,xstep,ystep,xpoints,ypoints, dxpoints, dypoints,dxlenth,dylenth, f,z0=377,hg=0.0008):
		self.k = 2 * math.pi * f / 3e8
		self.K_h = self.k / (4 * math.pi)
		self.K_e = -1j * self.k * z0 / (4 * math.pi)
		self.measure = Grid(xstep,ystep,xpoints,ypoints)
		self.rebuild = Grid(dxlenth/dxpoints,dylenth/dypoints,dxpoints,dypoints)
		self.A = np.zeros((3 * self.measure.total, 2 * self.rebuild.total), dtype=complex)
		self.T = np.zeros((4*self.measure.total,3*self.rebuild.total),dtype=complex)
		self.Q = np.zeros((6*self.measure.total,6*self.rebuild.total),dtype=complex)
		self.h = hg #重构点离地高度
		self.E = E
		self.H = H

		self.x = 0
		self.y = 0
		self.z = h
		self.r1 = 0
		self.r2 = 0
		self.G1r1 = 0
		self.G1r2 = 0
		self.G2r1 = 0
		self.G2r2 = 0
		self.G3r1 = 0
		self.G3r2 = 0

	def r(self):
		x = self.x
		y = self.y
		z = self.z
		h = self.h
		self.r1 = math.sqrt(x ** 2 + y ** 2 + (z - h) ** 2)
		self.r2 = math.sqrt(x ** 2 + y ** 2 + (z + h) ** 2)

	def G(self):
		k = self.k
		r1 = self.r1
		r2 = self.r2

		def f(r):
			return e**(-1j * k * r) / r

		def G1(r):
			return (3 / ((k * r) ** 2) + 1j * 3 / (k * r) - 1) * f(r)

		def G2(r):
			return (2 / ((k * r) ** 2 )+ 1j * 2 / (k * r)) * f(r)

		def G3(r):
			return (1/(k*r)+1j)*f(r)

		self.G1r1 = G1(r1)
		self.G1r2 = G1(r2)
		self.G2r1 = G2(r1)
		self.G2r2 = G2(r2)

	def T_build(self, Emax=1, Hmax=1):
		gamma_E = self.K_e
		gamma_H = self.K_h
		k = self.k
		measure = self.measure  # 测量点网格
		rebuild = self.rebuild  # 重建点网格
		T = np.zeros((4 * measure.total, 3 * rebuild.total), dtype=complex)

		if os.path.isfile(f"tmp\T_{self.z}_{Emax}_{Hmax}.npy"):
			self.T = np.load(f"tmp\T_{self.z}_{Emax}_{Hmax}.npy")
			if self.T.shape[1] == 3 * rebuild.total:
				return

		progressbar = ProgressBar(measure.total * rebuild.total)
		for M in range(measure.ypoints):
			for N in range(measure.xpoints):
				for m in range(rebuild.ypoints):
					for n in range(rebuild.xpoints):
						self.x = M * measure.xstep - n * rebuild.xstep
						self.y = N * measure.ystep - m * rebuild.ystep
						self.r()
						self.G()
						x = self.x
						y = self.y
						z = self.z
						d = self.h
						r1 = self.r1
						r2 = self.r2
						G1r1 = self.G1r1
						G2r2 = self.G2r2
						G1r2 = self.G1r2
						G2r1 = self.G2r1
						G3r1 = self.G3r1
						G3r2 = self.G3r2

						T_ExPz = gamma_E * ((z - d) * x / (r1 ** 2) * G1r1 +
											(z + d) * x / (r2 ** 2) * G1r2)
						T_ExMx = 0
						T_ExMy = gamma_E * ((z - d) / r1 * G3r1 +
											(z + d) / r2 * G3r2) * k
						T_EyPz = gamma_E * ((z - d) * y / (r1 ** 2) * G1r1 +
											(z + d) * y / (r2 ** 2) * G1r2)
						T_EyMx = gamma_E * (-(z - d) / r1 * G3r1
											-(z + d) / r2 * G3r2) * k
						T_EyMy = 0
						T_HxPz = -gamma_H * (y / r1 * G3r1 +
											 y / r2 * G3r2)
						T_HxMx = gamma_H * (-(y ** 2 + (z - d) ** 2) / (r1 ** 2) * G1r1 + G2r1
											-(y ** 2 + (z + d) ** 2) / (r2 ** 2) * G1r2 + G2r2) * k
						T_HxMy = gamma_H * (x * y / (r1 ** 2) * G1r1 +
											x * y / (r2 ** 2) * G1r2) * k
						T_HyPz = gamma_H * (x / r1 * G3r1 + x / r2 * G3r2)
						T_HyMx = gamma_H * (x * y / (r1 ** 2) * G1r1 +
											x * y / (r2 ** 2) * G1r2) * k
						T_HyMy = gamma_H * (-(x ** 2 + (z - d) ** 2) / (r1 ** 2) * G1r1 + G2r1
											-(x ** 2 + (z + d) ** 2) / (r2 ** 2) * G1r2 + G2r2) * k

						T[M * measure.xpoints + N][m * rebuild.xpoints + n] = T_ExPz/Emax
						T[M * measure.xpoints + N][m * rebuild.xpoints + n + rebuild.total] = T_ExMx/(Emax*k)
						T[M * measure.xpoints + N][m * rebuild.xpoints + n + rebuild.total * 2] = T_ExMy/(Emax*k)
						T[M * measure.xpoints + N + measure.total][m * rebuild.xpoints + n] = T_EyPz/Emax
						T[M * measure.xpoints + N + measure.total][m * rebuild.xpoints + n + rebuild.total] = T_EyMx/(Emax*k)
						T[M * measure.xpoints + N + measure.total][m * rebuild.xpoints + n + rebuild.total * 2] = T_EyMy/(Emax*k)
						T[M * measure.xpoints + N + measure.total * 2][m * rebuild.xpoints + n] = T_HxPz/Hmax
						T[M * measure.xpoints + N + measure.total * 2][m * rebuild.xpoints + n + rebuild.total] = T_HxMx/(Hmax*k)
						T[M * measure.xpoints + N + measure.total * 2][m * rebuild.xpoints + n + rebuild.total * 2] = T_HxMy/(Hmax*k)
						T[M * measure.xpoints + N + measure.total * 3][m * rebuild.xpoints + n] = T_HyPz/Hmax
						T[M * measure.xpoints + N + measure.total * 3][m * rebuild.xpoints + n + rebuild.total] = T_HyMx/(Hmax*k)
						T[M * measure.xpoints + N + measure.total * 3][m * rebuild.xpoints + n + rebuild.total * 2] = T_HyMy/(Hmax*k)
						progressbar.update()

		np.save(f"tmp\T_{self.z}_{Emax}_{Hmax}.npy", T)
		self.T = T

	def A_build(self):
			measure = self.measure  # 测量点网格
			rebuild = self.rebuild  # 重建点网格
			K_h = self.K_h
			k = self.k
			A = np.zeros((3 * measure.total, 2 * rebuild.total), dtype=complex)

			if os.path.isfile(f"tmp\A_{self.z}.npy"):
				self.A = np.load(f"tmp\A_{self.z}.npy")
				if self.A.shape[1] == 2 * rebuild.total:
					return
			progressbar = ProgressBar(measure.total*rebuild.total)
			for M in range(measure.ypoints):
				for N in range(measure.xpoints):
					for m in range(rebuild.ypoints):
						for n in range(rebuild.xpoints):
							self.x = M * measure.xstep - n * rebuild.xstep
							self.y = N * measure.ystep - m * rebuild.ystep
							self.r()
							self.G()
							x = self.x
							y = self.y
							z = self.z
							h = self.h
							r1 = self.r1
							r2 = self.r2
							G1r1 = self.G1r1
							G2r2 = self.G2r2
							G1r2 = self.G1r2
							G2r1 = self.G2r1
							A_HxMx = K_h * k * (-1 * (y ** 2 + (z - h) ** 2) / (r1 ** 2) * G1r1 + G2r1
												- 1 * (y ** 2 + (z + h) ** 2) / (r2 ** 2) * G1r2 + G2r2)
							A_HxMy = K_h * k * ((x * y) / (r1 ** 2) * G1r1
												+ (x * y) / (r2 ** 2) * G1r2)
							A_HyMx = K_h * k * ((x * y) / (r1 ** 2) * G1r1
												+ (x * y) / (r2 ** 2) * G1r2)
							A_HyMy = K_h * k * (-1 * ((z - h) ** 2 + x ** 2) / (r1 ** 2) * G1r1 + G2r1
												- 1 * ((z + h) ** 2 + x ** 2) / (r2 ** 2) * G1r2 + G2r2)
							A_HzMx = K_h * k * ((z - h) * x / r1 ** 2 * G1r1 + (z + h) * x / r2 ** 2 * G1r2)
							A_HzMy = K_h * k * ((z - h) * y / r1 ** 2 * G1r1 + (z + h) * y / r2 ** 2 * G1r2)

							A[M * measure.xpoints + N][m * rebuild.xpoints + n] = A_HxMx
							A[M * measure.xpoints + N][m * rebuild.xpoints + n + rebuild.total] = A_HxMy
							A[M * measure.xpoints + N + measure.total][m * rebuild.xpoints + n] = A_HyMx
							A[M * measure.xpoints + N + measure.total][m * rebuild.xpoints + n + rebuild.total] = A_HyMy
							A[M * measure.xpoints + N + measure.total * 2][m * rebuild.xpoints + n] = A_HzMx
							A[M * measure.xpoints + N + measure.total * 2][m * rebuild.xpoints + n + rebuild.total] = A_HzMy

							progressbar.update()

			self.A = A
			np.save(f"tmp\A_{self.z}",A)

	def Q_build(self, Emax=1, Hmax=1):
		gamma_E = self.K_e
		gamma_H = self.K_h
		k = self.k
		measure = self.measure  # 测量点网格
		rebuild = self.rebuild  # 重建点网格
		Q = np.zeros((6 * measure.total, 6 * rebuild.total), dtype=complex)

		if os.path.isfile(f"tmp\Q_{self.z}_{Emax}_{Hmax}.npy"):
			self.Q = np.load(f"tmp\Q_{self.z}_{Emax}_{Hmax}.npy")
			if self.Q.shape[1] == 6 * rebuild.total:
				return

		progressbar = ProgressBar(measure.total * rebuild.total)
		for M in range(measure.ypoints):
			for N in range(measure.xpoints):
				for m in range(rebuild.ypoints):
					for n in range(rebuild.xpoints):
						self.x = M * measure.xstep - n * rebuild.xstep
						self.y = N * measure.ystep - m * rebuild.ystep
						self.r()
						self.G()
						x = self.x
						y = self.y
						z = self.z
						d = self.h
						r1 = self.r1
						r2 = self.r2
						G1r1 = self.G1r1
						G2r2 = self.G2r2
						G1r2 = self.G1r2
						G2r1 = self.G2r1
						G3r1 = self.G3r1
						G3r2 = self.G3r2

						Q_ExPx = gamma_E*(-1*(y**2+(z-d)**2)/r1**2*G1r1+G2r1+(y**2+(z+d)**2)/r2**2*G1r2-G2r2)
						Q_ExPy = gamma_E*(x*y/(r1**2)*G1r1-x*y/r2**2*G1r2)
						Q_ExPz = gamma_E * ((z - d) * x / (r1 ** 2) * G1r1 +
											(z + d) * x / (r2 ** 2) * G1r2)
						Q_ExMx = 0
						Q_ExMy = gamma_E * ((z - d) / r1 * G3r1 +
											(z + d) / r2 * G3r2) * k
						Q_ExMz = gamma_E * (-1*y/r1*G3r1+
											y/r2*G3r2)*k

						Q_EyPx = gamma_E*(x*y/(r1**2)*G1r1-x*y/r2**2*G1r2)
						Q_EyPy = gamma_E*(-1*(x**2+(z-d)**2)/r1**2*G1r1+G2r1+(x**2+(z+d)**2)/r2**2*G1r2-G2r2)
						Q_EyPz = gamma_E * ((z - d) * y / (r1 ** 2) * G1r1 +
											(z + d) * y / (r2 ** 2) * G1r2)
						Q_EyMx = gamma_E * (-(z - d) / r1 * G3r1
											- (z + d) / r2 * G3r2) * k
						Q_EyMy = 0
						Q_EyMz = gamma_E * (x/r1*G3r1-x/r2*G3r2)*k

						Q_EzPx = gamma_E*((z-d)/(r1**2)*G1r1-(z+d)*x/r2**2*G1r2)
						Q_EzPy = gamma_E*(y*(z-d)/(r1**2)*G1r1-y*(z+d)/(r2**2)*G1r2)
						Q_EzPz = gamma_E * (-1*(x**2+y**2)/(r1**2)*G1r1+G2r1-1*(x**2+y**2)/(r2**2)*G1r2+G2r2)
						Q_EzMx = gamma_E*(y/r1*G3r1+y/r2*G3r2)*k
						Q_EzMy = -1*gamma_E*(x/r1*G3r1+x/r2*G3r2)*k
						Q_EzMz = 0

						Q_HxPx = 0
						Q_HxPy = gamma_H * ((z-d)/r1*G3r1-(z+d)/r2*G3r2)
						Q_HxPz = -gamma_H * (y / r1 * G3r1 +
											 y / r2 * G3r2)
						Q_HxMx = gamma_H * (-(y ** 2 + (z - d) ** 2) / (r1 ** 2) * G1r1 + G2r1
											- (y ** 2 + (z + d) ** 2) / (r2 ** 2) * G1r2 + G2r2) * k
						Q_HxMy = gamma_H * (x * y / (r1 ** 2) * G1r1 +
											x * y / (r2 ** 2) * G1r2) * k
						Q_HxMz = gamma_H * ((z - d)*x / (r1**2) * G1r1-(z+d)*x / (r2**2) * G1r2) * k

						Q_HyPx = gamma_H * (-(z-d)/r1*G3r1+(z+d)/r2*G3r2)
						Q_HyPy = 0
						Q_HyPz = gamma_H * (x / r1 * G3r1 + x / r2 * G3r2)
						Q_HyMx = gamma_H * (x * y / (r1 ** 2) * G1r1 +
											x * y / (r2 ** 2) * G1r2) * k
						Q_HyMy = gamma_H * (-(x ** 2 + (z - d) ** 2) / (r1 ** 2) * G1r1 + G2r1
											- (x ** 2 + (z + d) ** 2) / (r2 ** 2) * G1r2 + G2r2) * k
						Q_HyMz = gamma_H*(y*(z-d)/(r1**2)*G1r1-y*(z+d)/(r2**2)*G1r2)*k

						Q_HzPx = gamma_H*(y/r1*G3r1-y/r2*G3r2)
						Q_HzPy = gamma_H*(-x/r1*G3r1+x/r2*G3r2)
						Q_HzPz = 0
						Q_HzMx = gamma_H*((z-d)*x/(r1**2)*G1r1+(z+d)*x/(r2**2)*G1r2)*k
						Q_HzMy = gamma_H*((z-d)*y/(r1**2)*G1r1+(z+d)*y/(r2**2)*G1r2)*k
						Q_HzMz = gamma_H*(-(x**2+y**2)/(r1**2)*G1r1+G2r1+(x**2+y**2)/(r2**2)*G1r2-G2r2)*k


						Q[M * measure.xpoints + N][m * rebuild.xpoints + n] = Q_ExPx / Emax
						Q[M * measure.xpoints + N][m * rebuild.xpoints + n + rebuild.total] = Q_ExPy / Emax
						Q[M * measure.xpoints + N][m * rebuild.xpoints + n + rebuild.total * 2] = Q_ExPz / Emax
						Q[M * measure.xpoints + N][m * rebuild.xpoints + n + rebuild.total * 3] = Q_ExMx / (Emax*k)
						Q[M * measure.xpoints + N][m * rebuild.xpoints + n + rebuild.total * 4] = Q_ExMy / (Emax * k)
						Q[M * measure.xpoints + N][m * rebuild.xpoints + n + rebuild.total * 5] = Q_ExMz / (Emax * k)
						Q[M * measure.xpoints + N + measure.total][m * rebuild.xpoints + n] = Q_EyPx / Emax
						Q[M * measure.xpoints + N + measure.total][m * rebuild.xpoints + n + rebuild.total] = Q_EyPy / Emax
						Q[M * measure.xpoints + N + measure.total][m * rebuild.xpoints + n + rebuild.total * 2] = Q_EyPz / Emax
						Q[M * measure.xpoints + N + measure.total][m * rebuild.xpoints + n + rebuild.total * 3] = Q_EyMx / (Emax * k)
						Q[M * measure.xpoints + N + measure.total][m * rebuild.xpoints + n + rebuild.total * 4] = Q_EyMy / (Emax * k)
						Q[M * measure.xpoints + N + measure.total][m * rebuild.xpoints + n + rebuild.total * 5] = Q_EyMz / (Emax * k)
						Q[M * measure.xpoints + N + measure.total*2][m * rebuild.xpoints + n] = Q_EzPx / Emax
						Q[M * measure.xpoints + N + measure.total*2][
							m * rebuild.xpoints + n + rebuild.total] = Q_EzPy / Emax
						Q[M * measure.xpoints + N + measure.total*2][
							m * rebuild.xpoints + n + rebuild.total * 2] = Q_EzPz / Emax
						Q[M * measure.xpoints + N + measure.total*2][
							m * rebuild.xpoints + n + rebuild.total * 3] = Q_EzMx / (Emax * k)
						Q[M * measure.xpoints + N + measure.total*2][
							m * rebuild.xpoints + n + rebuild.total * 4] = Q_EzMy / (Emax * k)
						Q[M * measure.xpoints + N + measure.total*2][
							m * rebuild.xpoints + n + rebuild.total * 5] = Q_EzMz / (Emax * k)
						Q[M * measure.xpoints + N + measure.total*3][m * rebuild.xpoints + n] = Q_HxPx / Hmax
						Q[M * measure.xpoints + N + measure.total*3][
							m * rebuild.xpoints + n + rebuild.total*3] = Q_HxPy / Hmax
						Q[M * measure.xpoints + N + measure.total*3][
							m * rebuild.xpoints + n + rebuild.total * 2] = Q_HxPz / Hmax
						Q[M * measure.xpoints + N + measure.total*3][
							m * rebuild.xpoints + n + rebuild.total * 3] = Q_HxMx / (Hmax * k)
						Q[M * measure.xpoints + N + measure.total*3][
							m * rebuild.xpoints + n + rebuild.total * 4] = Q_HxMy / (Hmax * k)
						Q[M * measure.xpoints + N + measure.total*3][
							m * rebuild.xpoints + n + rebuild.total * 5] = Q_HxMz / (Hmax * k)
						Q[M * measure.xpoints + N + measure.total*4][m * rebuild.xpoints + n] = Q_HyPx / Hmax
						Q[M * measure.xpoints + N + measure.total*4][
							m * rebuild.xpoints + n + rebuild.total*3] = Q_HyPy / Hmax
						Q[M * measure.xpoints + N + measure.total*4][
							m * rebuild.xpoints + n + rebuild.total * 2] = Q_HyPz / Hmax
						Q[M * measure.xpoints + N + measure.total*4][
							m * rebuild.xpoints + n + rebuild.total * 3] = Q_HyMx / (Hmax * k)
						Q[M * measure.xpoints + N + measure.total*4][
							m * rebuild.xpoints + n + rebuild.total * 4] = Q_HyMy / (Hmax * k)
						Q[M * measure.xpoints + N + measure.total*4][
							m * rebuild.xpoints + n + rebuild.total * 5] = Q_HyMz / (Hmax * k)
						Q[M * measure.xpoints + N + measure.total*5][m * rebuild.xpoints + n] = Q_HzPx / Hmax
						Q[M * measure.xpoints + N + measure.total*5][
							m * rebuild.xpoints + n + rebuild.total*3] = Q_HzPy / Hmax
						Q[M * measure.xpoints + N + measure.total*5][
							m * rebuild.xpoints + n + rebuild.total * 2] = Q_HzPz / Hmax
						Q[M * measure.xpoints + N + measure.total*5][
							m * rebuild.xpoints + n + rebuild.total * 3] = Q_HzMx / (Hmax * k)
						Q[M * measure.xpoints + N + measure.total*5][
							m * rebuild.xpoints + n + rebuild.total * 4] = Q_HzMy / (Hmax * k)
						Q[M * measure.xpoints + N + measure.total*5][
							m * rebuild.xpoints + n + rebuild.total * 5] = Q_HzMz / (Hmax * k)
						progressbar.update()

		np.save(f"tmp\Q_{self.z}_{Emax}_{Hmax}.npy", Q)
		self.Q = Q

	def solve(self,state="A", scope=0.001):
		if state == "A":
			self.A_build()
			A = self.A
			H = self.H

			# U,S,V = np.linalg.svd(A)
			# VT=V.T
			# ambda = scope * S[0]
			# S_ = np.diag([1 / (xi ** 2 + ambda ** 2) for xi in S])
			# X = VT @ S_ @ V @ A.T @ H
			# def func(X):
			# 	F = np.zeros(20)
			# 	for i in range(20):
			# 		D = np.zeros(2*self.rebuild.total,dtype=complex)
			# 		for j in range(2*self.rebuild.total):
			# 			D[j]=X[i][j*2]+1j*X[i][j*2+1]
			# 		H_dipole = A @ D
			# 		f = np.linalg.norm(H-H_dipole)
			# 		F[i] = f
			#
			# 	return F
			# opt = ps.single.GlobalBestPSO(n_particles=20,dimensions=4*self.rebuild.total, options={'c1': 0.4, 'c2': 0.4, 'w': 0.2})
			# cost,pos = opt.optimize(func,3000)
			# X = np.zeros(2 * self.rebuild.total, dtype=complex)
			# for j in range(2 * self.rebuild.total):
			# 	X[j] = pos[j * 2] + 1j * pos[j * 2 + 1]

			def func(X):
				D = np.zeros(2 * self.rebuild.total, dtype=complex)
				for j in range(2*self.rebuild.total):
					D[j]=X[j*2]+1j*X[j*2+1]
				H_dipole = A @ D
				f = np.mean(np.abs(np.angle(H_dipole)-np.angle(H)))*1+np.linalg.norm(H_dipole-H)*0
				print('\r', f, end='')
				return f

			de = DE(func=func, n_dim=4*self.rebuild.total, size_pop=50, max_iter=1000)
			if os.path.exists("output/A_DE.npy"):
				D = np.load("output/A_DE.npy")
			else:
				D = np.linalg.lstsq(A, H, rcond=None)[0]
			for j in range(2 * self.rebuild.total):
				for i in range(50):
					de.X[i][j*2] = np.real(D[j])*(1 + 5*random.random())
					de.X[i][j*2+1] = np.imag(D[j])*(1 + 5*random.random())
			X,_ = de.run()
			D = np.zeros(2 * self.rebuild.total, dtype=complex)
			for j in range(2 * self.rebuild.total):
				D[j] = X[j * 2] + 1j * X[j * 2 + 1]
			# X = np.linalg.lstsq(A,H,rcond=None)[0]
			np.save("output/A_DE.npy", D)
			return D
		elif state == "Q":
			Emax = np.max(np.abs(self.E))
			Hmax = np.max(np.abs(self.H))

			self.Q_build(Emax=Emax, Hmax=Hmax)
			Q=self.Q
			Y = np.append(self.E/Emax,self.H/Hmax)
			X = np.linalg.lstsq(Q, Y, rcond=None)[0]
			return X
		else:	#解出来的不直接是dipole，Mx,My需要乘一个k
			Emax = np.max(np.abs(self.E[:self.measure.total * 2]))
			Hmax = np.max(np.abs(self.H[:self.measure.total * 2]))

			self.T_build(Emax=Emax,Hmax=Hmax)
			T = self.T
			Y = np.append(self.E[:self.measure.total*2]/Emax,self.H[:self.measure.total*2]/Hmax)
			# U, S, V = np.linalg.svd(T)
			# VT = V.T
			# ambda = scope * S[0]
			# S_ = np.diag([1 / (xi ** 2 + ambda ** 2) for xi in S])
			# X = VT @ S_ @ V @ T.T @ Y
			X = np.linalg.lstsq(T, Y, rcond=None)[0]
			return X





if __name__ == "__main__":
	print(e)


