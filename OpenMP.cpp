#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <windows.h>

using namespace std;

bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    int limit = static_cast<int>(sqrt(n));
    for (int i = 3; i <= limit; i += 2)
        if (n % i == 0) return false;
    return true;
}

int main() {
    omp_set_num_threads(4);

    const int RANGE_START = 1;
    const int RANGE_END = 10000000;

    double start_time = omp_get_wtime();

    int num_threads = 0;
    vector<vector<int>> primes_per_thread;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp single
        {
            num_threads = omp_get_num_threads();
            primes_per_thread.resize(num_threads);
        }

        vector<int>& local = primes_per_thread[tid];

#pragma omp for schedule(dynamic)
        for (int i = RANGE_START; i <= RANGE_END; ++i) {
            if (is_prime(i)) {
                local.push_back(i);
            }
        }
    }

    // Об'єднання всіх результатів
    vector<int> all_primes;
    for (int t = 0; t < num_threads; ++t) {
        all_primes.insert(all_primes.end(),
            primes_per_thread[t].begin(),
            primes_per_thread[t].end());
    }

    double end_time = omp_get_wtime();

    cout << "Threads used: " << num_threads << endl;
    cout << "Range: [" << RANGE_START << ", " << RANGE_END << "]" << endl;
    cout << "Total primes found: " << all_primes.size() << endl;
    cout << "Execution time: " << (end_time - start_time) << " seconds" << endl;

    return 0;
}
