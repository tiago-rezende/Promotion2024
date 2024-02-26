from mpi4py import MPI  # Importing MPI for parallel processing
import numpy as np

# Duplicate the communicator to create a new one
comm = MPI.COMM_WORLD.Dup()
rank = comm.Get_rank()  # Get the rank of the current process
nbp = comm.Get_size()   # Get the total number of processes

# Class to group numbers
class NumberGrouper:
    def __init__(self, numbers):
        self.numbers = numbers

    def get_numbers(self):
        return self.numbers

    def group_numbers(self):
        # Initialize an empty dictionary to group numbers
        grouped_numbers = {}
        # Iterate over the sorted numbers
        for num in sorted(self.numbers):
            group = int(num)
            if group not in grouped_numbers:
                grouped_numbers[group] = []  # Using an empty list
            grouped_numbers[group].append(num)
        return grouped_numbers

# Root process (rank 0) operations
if rank == 0:
    low = 0
    high = 99
    size = 10  # Size of the vector
    interval = high - low
    N = interval // nbp
    # Generate random numbers without replacement
    random_numbers = np.random.choice(np.arange(low, high), size, replace=False)

    print("Vector of random values:", random_numbers)

    localvetors = {}
    # Initialize localvectors dictionary for each process
    for i in range(0, nbp):
        localvetors[i] = []

    # Divide the random numbers into buckets for each process
    for i in range(0, nbp):
        for num in random_numbers:  
            # Assign numbers to the corresponding bucket
            if low + N * i <= num < low + N * (i + 1):
                localvetors[i].append(num)

    # Send arrays to each process except for rank 0
    print("Vector 0 send", localvetors[0])
    for i in range(nbp):
        if i > 0:
            comm.send(localvetors[i], dest=i)
            print("Vector", i, "send", localvetors[i])

    received_data = {}

    # Process the numbers for rank 0
    ordered = NumberGrouper(localvetors[0])
    grouped_numbers = ordered.group_numbers()
    keys_list = list(grouped_numbers.keys())
    print("Vector 0 grouped", keys_list)

    # Receive and process the numbers from other processes
    for i in range(1, nbp):
        received_data[i] = []
        received_data[i] = comm.recv(source=i)
        print("Vector", i, "grouped", received_data[i])

    # Combine the final results
    final = keys_list
    for i in range(1, nbp):
        final.extend(received_data[i])
    print("Final Vector", final)

# Non-root processes
else:
    # Receive data from rank 0
    local = comm.recv(source=0)
    ordered = NumberGrouper(local)
    grouped_numbers = ordered.group_numbers()
    keys_list = list(grouped_numbers.keys())
    comm.send(keys_list, dest=0)
