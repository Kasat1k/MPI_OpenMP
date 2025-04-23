#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <windows.h>

using namespace std;

// Перевірка, чи є число простим
bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    int limit = static_cast<int>(sqrt(n));
    for (int i = 3; i <= limit; i += 2)
        if (n % i == 0) return false;
    return true;
}

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Межі пошуку
    const int RANGE_START = 1;
    const int RANGE_END = 10000000;

    double start_time = MPI_Wtime();

    // Ділимо діапазон на піддіапазони
    int total_numbers = RANGE_END - RANGE_START + 1;
    int chunk_size = total_numbers / size;
    int remainder = total_numbers % size;

    int local_start = RANGE_START + rank * chunk_size + min(rank, remainder);
    int local_end = local_start + chunk_size - 1;
    if (rank < remainder) local_end++;

    vector<int> local_primes;

    // Пошук простих чисел у локальному діапазоні
    for (int i = local_start; i <= local_end; ++i) {
        if (is_prime(i)) {
            local_primes.push_back(i);
        }
    }

    // Збір результатів на rank 0
    int local_count = static_cast<int>(local_primes.size());
    vector<int> all_counts(size);

    MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> all_primes;
    vector<int> displs;

    if (rank == 0) {
        int total = 0;
        displs.push_back(0);
        for (int i = 0; i < size; ++i) {
            total += all_counts[i];
            if (i > 0)
                displs.push_back(displs[i - 1] + all_counts[i - 1]);
        }
        all_primes.resize(total);
    }

    MPI_Gatherv(local_primes.data(), local_count, MPI_INT,
        all_primes.data(), all_counts.data(), displs.data(), MPI_INT,
        0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        cout << "Total primes found: " << all_primes.size() << endl;
        cout << "Execution time: " << (end_time - start_time) << " seconds" << endl;
        // Якщо хочеш — можеш розкоментувати вивід чисел:
        // for (int p : all_primes) cout << p << " ";
        // cout << endl;
    }

    MPI_Finalize();
    return 0;
}
