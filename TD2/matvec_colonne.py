from mpi4py import MPI

comm = MPI.COMM_WORLD.Dup()
rank = comm.Get_rank()
nbp = comm.Get_size()

# Matrix-vector product v = A.u
import numpy as np

# Problem dimension (can be changed)
dim = 120
N = dim // nbp
remaining_cols = dim % nbp
if rank < remaining_cols:
    N += 1

# Initialize the matrix A
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
print(f"A = {A}")

# Initialize the vector u
u = np.array([i+1. for i in range(dim)])
print(f"u = {u}")

# Initialize the vector v
v = np.zeros(dim)

# Scatter the matrix A
A_local = np.zeros((N, dim), dtype=np.float64)
comm.Scatter(A, A_local, root=0)

# Compute the local matrix-vector product
v_local = np.zeros(N)
for j in range(dim):
    for i in range(N):
    
        v_local[i] += A_local[i][j] * u[j]

# Reduce the local results to the vector v at the root process
comm.Allgather([v_local, MPI.DOUBLE], [v, MPI.DOUBLE])

# Print the vector v at the root process
if rank == 0:
    print(f"Process {rank} received the vector v = {v}")
