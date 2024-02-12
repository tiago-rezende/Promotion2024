# Calcul de l'ensemble de Mandelbrot en python
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

comm = MPI.COMM_WORLD.Dup()
rank = comm.Get_rank()
nbp = comm.Get_size()

@dataclass

class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height


convergence = np.empty((width, height), dtype=np.double)
# Calcul de l'ensemble de mandelbrot :
deb = time()


if rank == 0:  # rank == 0 => master
    glob_array = np.empty((height, width), dtype=np.double)
    count_task = 0
    for i in range(1, nbp):
        comm.send(count_task, dest=i)
        count_task = 1 +count_task
        
    while count_task + 1 < height:
        Status = MPI.Status()
        y, ProcesVector = comm.recv(source=MPI.ANY_SOURCE, status=Status)
        
        for x in range(height):
            glob_array[y, x] = ProcesVector[x]
        
        comm.send(count_task, dest=Status.Get_source())
        count_task = 1 +count_task
        source_rank = Status.Get_source()

    for i in range(1, nbp):
        comm.send(None, dest=i) 
        
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(glob_array) * 255))
    image.show()

if rank > 0:
    deb = time()
    while True:

        y = comm.recv(source=0)
        if y is None:  
            break

        ProcesVector = np.empty(width, dtype=np.double)
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
            ProcesVector[x] = np.array(mandelbrot_set.convergence(complex(-2.0 + scaleX * x, -1.125 + scaleY * y)))
            
        comm.send((y, ProcesVector), dest=0)
    fin=time()
    print("Temps du processus",rank,":",fin-deb)



