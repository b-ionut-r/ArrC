//
// Created by Bujor Ionut Raul on 16.11.2025.
//


#include <iostream>
#include "ndarray.cuh"
using namespace std;

int main() {
    cout << "Enter the size of the matrix: ";
    int n; cin >> n;
    NDArray<float> A({n, n}), B({n, n}), C({n, n});
    cout << "Enter the first matrix: " << endl;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cin >> A[{i, j}];
        }
        cout << endl;
    }
    cout << "Enter the second matrix: " << endl;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            cin >> B[{i, j}];
        }
        cout << endl;
    }
    C = A + B;
    cout << "The sum is: " << endl;
    cout << C << "\n\n";
    NDArray<float> D = C[{Slice(0, 3, 2), Slice(0, 3, 2)}]; // view colturi
    cout << D << "\n\n";
    NDArray<float> ZEROS({2, 2});
    ZEROS = 0;
    D = ZEROS;
    cout << D << "\n\n";
    cout << C << "\n\n";
    return 0;

}

