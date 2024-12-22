import kagglehub
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random

path = kagglehub.dataset_download("chrico03/art-garfunkels-library")
print("Path to dataset files:", path)

dataset_file = f"{path}/Art Garfunkel Library.csv"
data = pd.read_csv(dataset_file)

print(data.columns)

books = data['Books'].dropna().sort_values().values

books = np.array(books)
print("Jumlah buku:", len(books))

def binary_search(arr, target):
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

def jump_search(arr, target):
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

def simulate_search(books, search_algo):
    target = random.choice(books)

    start_time = time.time()
    search_algo(books, target)
    return time.time() - start_time

sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000, 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000, 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000]

binary_times_samples = []
jump_times_samples = []
dataset_sizes = []

for size in sizes:
    subset_books = books[:size]

    for _ in range(100):
        binary_search_time = simulate_search(subset_books, binary_search)
        binary_times_samples.append(binary_search_time)

        jump_search_time = simulate_search(subset_books, jump_search)
        jump_times_samples.append(jump_search_time)

        dataset_sizes.append(size)

df = pd.DataFrame({
    'Ukuran Dataset': dataset_sizes,
    'Waktu Binary Search (detik)': binary_times_samples,
    'Waktu Jump Search (detik)': jump_times_samples
})

average_times = df.groupby('Ukuran Dataset').mean().reset_index()

print(average_times)

plt.figure(figsize=(15, 9))
plt.plot(average_times['Ukuran Dataset'], average_times['Waktu Binary Search (detik)'], 
         label='Binary Search', color='blue', marker='o')
plt.plot(average_times['Ukuran Dataset'], average_times['Waktu Jump Search (detik)'], 
         label='Jump Search', color='red', marker='x')

plt.xlabel('Ukuran Dataset (Jumlah Buku)', fontsize=12)
plt.ylabel('Waktu Pencarian (detik)', fontsize=12)
plt.title('Perbandingan Waktu Pencarian: Binary Search vs Jump Search (Rata-rata)', fontsize=14)
plt.legend()

plt.grid(True)
plt.xscale('linear')
plt.yscale('log')
plt.show()