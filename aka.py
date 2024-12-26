import kagglehub
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from google.colab import data_table

data_table.enable_dataframe_formatter()

path = kagglehub.dataset_download("chrico03/art-garfunkels-library")
print("Path to dataset files:", path)

dataset_file = f"{path}/Art Garfunkel Library.csv"
data = pd.read_csv(dataset_file)

books = data['Books'].dropna().sort_values().values
books = np.array(books)
print("Jumlah buku:", len(books))

def binary_search_iterative(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def jump_search_iterative(arr, target):
    n = len(arr)
    step = int(np.sqrt(n))
    prev = 0
    while arr[min(step, n)-1] < target:
        prev = step
        step += int(np.sqrt(n))
        if prev >= n:
            return -1
    for i in range(prev, min(step, n)):
        if arr[i] == target:
            return i
    return -1

def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

def jump_search_recursive(arr, target, step, prev):
    n = len(arr)
    if prev >= n:
        return -1
    if arr[min(step, n)-1] < target:
        return jump_search_recursive(arr, target, step + int(np.sqrt(n)), step)
    for i in range(prev, min(step, n)):
        if arr[i] == target:
            return i
    return -1

def simulate_search(books, search_algo, *args):
    target = random.choice(books)
    start_time = time.time()
    search_algo(books, target, *args)
    return time.time() - start_time

sizes = list(range(1, 151, 5))

binary_iterative_times_samples = []
binary_recursive_times_samples = []
jump_iterative_times_samples = []
jump_recursive_times_samples = []
dataset_sizes = []

for size in sizes:
    subset_books = books[:size]

    for _ in range(20):
        binary_iterative_search_time = simulate_search(subset_books, binary_search_iterative)
        binary_iterative_times_samples.append(binary_iterative_search_time)

        binary_recursive_search_time = simulate_search(subset_books, binary_search_recursive, 0, len(subset_books) - 1)
        binary_recursive_times_samples.append(binary_recursive_search_time)

        jump_iterative_search_time = simulate_search(subset_books, jump_search_iterative)
        jump_iterative_times_samples.append(jump_iterative_search_time)

        jump_recursive_search_time = simulate_search(subset_books, jump_search_recursive, int(np.sqrt(size)), 0)
        jump_recursive_times_samples.append(jump_recursive_search_time)

        dataset_sizes.append(size)

df = pd.DataFrame({
    'Ukuran Dataset': dataset_sizes,
    'Waktu Binary Search Iteratif (detik)': binary_iterative_times_samples,
    'Waktu Binary Search Rekursif (detik)': binary_recursive_times_samples,
    'Waktu Jump Search Iteratif (detik)': jump_iterative_times_samples,
    'Waktu Jump Search Rekursif (detik)': jump_recursive_times_samples
})

average_times = df.groupby('Ukuran Dataset').mean().reset_index()

average_times

plt.figure(figsize=(15, 9))
plt.plot(average_times['Ukuran Dataset'], average_times['Waktu Binary Search Iteratif (detik)'], 
         label='Binary Search Iteratif', color='blue', marker='o')
plt.plot(average_times['Ukuran Dataset'], average_times['Waktu Binary Search Rekursif (detik)'], 
         label='Binary Search Rekursif', color='cyan', marker='x')
plt.plot(average_times['Ukuran Dataset'], average_times['Waktu Jump Search Iteratif (detik)'], 
         label='Jump Search Iteratif', color='red', marker='o')
plt.plot(average_times['Ukuran Dataset'], average_times['Waktu Jump Search Rekursif (detik)'], 
         label='Jump Search Rekursif', color='magenta', marker='x')

plt.xlabel('Ukuran Dataset (Jumlah Buku)', fontsize=12)
plt.ylabel('Waktu Pencarian (detik)', fontsize=12)
plt.title('Perbandingan Waktu Pencarian: Binary Search vs Jump Search (Rata-rata)', fontsize=14)
plt.legend()

plt.grid(True)
plt.xscale('linear')
plt.yscale('log')
plt.show()
