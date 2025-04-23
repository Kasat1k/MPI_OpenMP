#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

void print_matrix(const vector<vector<double>>& A) {
    for (const auto& row : A) {
        for (double val : row)
            cout << val << "\t";
        cout << endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 4; // Розмір матриці
    vector<vector<double>> A(N, vector<double>(N));

    if (rank == 0) {
        // Вхідна матриця
        A = {
            {2, -1, 0, 3},
            {1, 0, 4, 2},
            {0, 1, 2, -1},
            {3, 2, 1, 0}
        };
        cout << "Початкова матриця:" << endl;
        print_matrix(A);
    }

    // Розсилка матриці
    for (int i = 0; i < N; ++i)
        MPI_Bcast(A[i].data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int swaps = 0;

    for (int k = 0; k < N - 1; ++k) {
        if (rank == 0) {
            // Частковий вибір головного елемента
            int max_row = k;
            double max_val = fabs(A[k][k]);
            for (int i = k + 1; i < N; ++i) {
                if (fabs(A[i][k]) > max_val) {
                    max_val = fabs(A[i][k]);
                    max_row = i;
                }
            }

            if (max_row != k) {
                swap(A[k], A[max_row]);
                swaps++;
            }
        }

        // Розсилка оновленого рядка
        MPI_Bcast(A[k].data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Розподіл обчислень по процесах
        for (int i = k + 1 + rank; i < N; i += size) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < N; ++j) {
                A[i][j] -= factor * A[k][j];
            }
        }

        // Синхронізація
        for (int i = k + 1; i < N; ++i) {
            MPI_Bcast(A[i].data(), N, MPI_DOUBLE, (i - (k + 1)) % size, MPI_COMM_WORLD);
        }
    }

    // Обчислення визначника (на rank 0)
    if (rank == 0) {
        double det = (swaps % 2 == 0 ? 1 : -1);
        for (int i = 0; i < N; ++i) {
            det *= A[i][i];
        }

        cout << "\nТрикутна матриця після Гаусса:" << endl;
        print_matrix(A);
        cout << "\nВизначник матриці: " << det << endl;
    }

    MPI_Finalize();
    return 0;
}
