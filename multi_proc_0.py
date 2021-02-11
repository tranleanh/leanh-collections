import multiprocessing as mp
from multiprocessing import Pool
import time

# print("Number of processors: ", mp.cpu_count())

def Process1(input_):
    time.sleep(30)
    a = input_**2
    return a

def Process2(input_):
    time.sleep(30)
    b = input_**2
    return b

def Process3(input_):
    time.sleep(30)
    c = input_**2
    return c

def FinalProcess(a,b,c):
    final_results = a + b + c
    return final_results


def run_parallel():

    print("Run Parallel...")

    start_time = time.time()
    pool = Pool(processes=3)

    value1 = pool.apply_async(Process1, [1])
    value2 = pool.apply_async(Process2, [2])
    value3 = pool.apply_async(Process3, [3])

    output1 = value1.get()
    output2 = value2.get()
    output3 = value3.get()

    print(output1, output2, output3)

    process_time = time.time() - start_time
    print(f"Parallel: {process_time} seconds")


def run_1cpu():

    print("Run on 1 CPU...")

    start_time = time.time()
    output1 = Process1(1)
    output2 = Process2(2)
    output3 = Process3(3)

    print(output1, output2, output3)

    process_time = time.time() - start_time
    print(f"1 CPU: {process_time} seconds")


def main():
    run_parallel()
    run_1cpu()


if __name__ == "__main__":
    main()
