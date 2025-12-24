//
// ArrC - Framework pentru Deep Learning cu CUDA
// Proiect POO - Bujor Ionut Raul
//

#include <iostream>
#include <algorithm>
#include <sstream>
#include "include/ndarray.cuh"
#include "include/tensor.h"
#include "include/functions/arithmetic.h"
#include "include/optim/sgd.cuh"
#include "include/optim/adam.cuh"
#include "include/optim/rmsprop.cuh"
#include "include/exceptions.h"

using namespace std;

// Functie ce primeste pointer la clasa de baza (UPCAST)
void optimizeStep(Optimizer* opt) {
    cout << "  Optimizer step via base class pointer" << endl;
    cout << "  LR = " << opt->getLR() << endl;
    opt->step();  // apel polimorfic
    cout << "  t = " << opt->getT() << endl;
}

// Functie ce face DOWNCAST cu dynamic_cast
void showOptimizerDetails(Optimizer* opt) {
    // Incercam downcast la SGD
    if (SGD* sgd = dynamic_cast<SGD*>(opt)) {
        cout << "  [DOWNCAST] Este SGD, beta = " << sgd->getBeta() << endl;
        cout << "  " << *sgd;
        return;
    }
    // Incercam downcast la Adam
    if (Adam* adam = dynamic_cast<Adam*>(opt)) {
        cout << "  [DOWNCAST] Este Adam, beta1 = " << adam->getBeta1() << endl;
        cout << "  " << *adam;
        return;
    }
    // Incercam downcast la RMSProp
    if (RMSProp* rms = dynamic_cast<RMSProp*>(opt)) {
        cout << "  [DOWNCAST] Este RMSProp, eps = " << rms->getEps() << endl;
        cout << "  " << *rms;
        return;
    }
}

// Functie care arunca exceptie pentru demonstrarea propagarii
void innerFunction() {
    throw ShapeMismatchException("Eroare la nivel inner");
}
void outerFunction() {
    innerFunction();  // exceptia se propaga
}

void meniu() {

    cout << "\n===== ArrC - Deep Learning Framework =====\n";
    cout << "1. Demo NDArray (template, operatori, copiere)\n";
    cout << "2. Demo Tensor + Autograd (backward)\n";
    cout << "3. Demo Optimizers (polimorfism, upcast/downcast)\n";
    cout << "4. Demo Exceptii (ierarhie, propagare, catch upcast)\n";
    cout << "5. Demo Training (optimizare functie)\n";
    cout << "6. Info framework\n";
    cout << "0. Exit\n";
    cout << "Alegere: ";
}

int main() {
    int opt;

    while (true) {
        meniu();
        cin >> opt;

        if (opt == 0) {
            cout << "La revedere!\n";
            break;
        }

        switch (opt) {
        case 1: {
            // ===== DEMO NDARRAY =====
            cout << "\n--- Demo NDArray ---\n";

            // Template cu 2 tipuri diferite
            NDArray<float> arr1({2, 2});   // instantiere 1
            NDArray<double> arr2({2, 2});  // instantiere 2
            arr1 = 5.0f;
            arr2 = 3.0;
            cout << "NDArray<float>:\n" << arr1 << endl;
            cout << "NDArray<double>:\n" << arr2 << endl;

            // Copy constructor
            NDArray<float> copie(arr1);
            cout << "Copie (copy constructor):\n" << copie << endl;

            // operator= copiere
            NDArray<float> atribuit({2, 2});
            atribuit = arr1;
            cout << "Atribuit (operator=):\n" << atribuit << endl;

            // Operatori aritmetici (membri)
            NDArray<float> b({2, 2});
            b = 2.0f;
            cout << "arr1 + b:\n" << (arr1 + b) << endl;
            cout << "arr1 * b:\n" << (arr1 * b) << endl;

            // Operator non-membru (scalar + array)
            cout << "10 + arr1:\n" << (10.0f + arr1) << endl;
            cout << "100 / arr1:\n" << (100.0f / arr1) << endl;

            // operator>> (citire)
            NDArray<float> citit({2, 2});
            stringstream ss("1 2 3 4");
            ss >> citit;
            cout << "Citit din stringstream:\n" << citit << endl;

            // operator[] pentru indexare
            citit[{0, 0}] = 99.0f;
            cout << "Dupa modificare [0,0]=99:\n" << citit << endl;

            // Static member
            cout << "Memorie GPU alocata: " << NDArray<float>::getTotalAllocatedMemory() << " bytes\n";
            break;
        }

        case 2: {
            // ===== DEMO TENSOR + AUTOGRAD =====
            cout << "\n--- Demo Tensor & Autograd ---\n";

            // Handle-style Tensor API (shared_ptr internally)
            Tensor<float> x = tensor::zeros<float>({1}, true);
            Tensor<float> y = tensor::zeros<float>({1}, true);
            x.data()[{0}] = 3.0f;
            y.data()[{0}] = 4.0f;

            cout << "x = " << x.data() << endl;
            cout << "y = " << y.data() << endl;

            // Construim graf: z = x * y
            auto z = x * y;
            cout << "z = x * y = " << z.data() << endl;

            // Backward pass
            z.backward();
            cout << "dz/dx = " << *(x.grad()) << " (ar trebui 4)\n";
            cout << "dz/dy = " << *(y.grad()) << " (ar trebui 3)\n";

            // No global registry - Functions are owned by Tensors

            // No manual delete needed - shared_ptr handles cleanup
            break;
        }

        case 3: {
            // ===== DEMO OPTIMIZERS (POLIMORFISM) =====
            cout << "\n--- Demo Optimizers ---\n";

            // Cream un tensor pentru parametri (handle-style API)
            Tensor<float> w = tensor::ones<float>({2, 2}, true);
            *w.grad() = 0.1f;

            vector<tensor::TensorSharedVariant> params = {w.shared()};

            // Strategy Pattern - diferiti optimizeri
            SGD sgd(params, 0.01f, 0.0f, 0.9f, FLOAT);
            Adam adam(params, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8, FLOAT);
            RMSProp rms(params, 0.01f, 0.0f, 0.99f, 1e-8, FLOAT);

            // Vector de pointeri la clasa de baza (UPCAST implicit)
            vector<Optimizer*> optimizers = {&sgd, &adam, &rms};

            cout << "Apelam step() prin base class pointer:\n";
            for (Optimizer* o : optimizers) {
                o->zeroGrad();
                *w.grad() = 0.1f;
                optimizeStep(o);  // UPCAST
                cout << endl;
            }

            // DOWNCAST cu dynamic_cast
            cout << "Inspectam cu dynamic_cast:\n";
            for (Optimizer* o : optimizers) {
                showOptimizerDetails(o);  // DOWNCAST
            }

            // Virtual destructor demo
            cout << "\nVirtual destructor test:\n";
            Tensor<float> tmp = tensor::ones<float>({1}, true);
            *tmp.grad() = 0.1f;
            vector<tensor::TensorSharedVariant> tmp_params = {tmp.shared()};

            Optimizer* ptr = new Adam(tmp_params, 0.001f, 0.0f, 0.9f, 0.999f);
            delete ptr;  // virtual destructor asigura cleanup corect
            cout << "Sters prin Optimizer* - OK\n";
            break;
        }

        case 4: {
            // ===== DEMO EXCEPTII =====
            cout << "\n--- Demo Exceptii ---\n";

            // Ierarhie: std::exception -> ArrcException -> derived

            // 1. Catch specific
            try {
                throw IndexingException("index invalid");
            } catch (const IndexingException& e) {
                cout << "Caught specific: " << e.what() << endl;
            }

            // 2. Catch prin clasa de baza (UPCAST in catch)
            cout << "\nCatch prin ArrcException& (upcast):\n";
            try {
                throw SizeMismatchException("marimi diferite");
            } catch (const ArrcException& e) {
                cout << "  Caught as ArrcException: " << e.what() << endl;
            }

            try {
                throw CudaKernelException("CUDA error");
            } catch (const ArrcException& e) {
                cout << "  Caught as ArrcException: " << e.what() << endl;
            }

            // 3. Propagare exceptii
            cout << "\nPropagare exceptii prin stack:\n";
            try {
                outerFunction();  // arunca din innerFunction
            } catch (const ArrcException& e) {
                cout << "  Caught la top level: " << e.what() << endl;
            }

            // 4. Exceptie reala din NDArray
            cout << "\nExceptie reala din framework:\n";
            try {
                NDArray<float> a({2, 3});
                NDArray<float> b({3, 2});
                a = b;  // shape mismatch!
            } catch (const ShapeMismatchException& e) {
                cout << "  " << e.what() << endl;
            }

            try {
                NDArray<float> a({2, 2});
                float x = a[{0}];  // nevoie de 2 indici
                (void)x;
            } catch (const ArrcException& e) {
                cout << "  " << e.what() << endl;
            }
            break;
        }

        case 5: {
            // ===== DEMO TRAINING =====
            cout << "\n--- Demo Training ---\n";
            cout << "Minimizam f(x) = (x - 3)^2, start x = 10\n\n";

            Tensor<float> x = tensor::zeros<float>({1}, true);
            x.data()[{0}] = 10.0f;

            Tensor<float> target = tensor::zeros<float>({1}, false);
            target.data()[{0}] = 3.0f;

            vector<tensor::TensorSharedVariant> params = {x.shared()};
            Adam optimizer(params, 0.3f, 0.0f, 0.9f, 0.999f);

            for (int step = 0; step < 30; step++) {
                optimizer.zeroGrad();

                // Forward: loss = (x - target)^2
                auto diff = x - target;
                auto loss = diff * diff;

                float loss_val = loss.data()[{0}];
                if (step % 5 == 0) {
                    cout << "Step " << step << ": x = " << x.data()[{0}]
                         << ", loss = " << loss_val << endl;
                }
                loss.backward();
                optimizer.step();

                // No manual delete needed - shared_ptr handles cleanup
            }
            cout << "\nFinal: x = " << x.data()[{0}] << " (target: 3.0)\n";
            break;
        }

        case 6: {
            // ===== INFO FRAMEWORK =====
            cout << "\n--- Info Framework ---\n";
            cout << "Ierarhii de clase:\n";
            cout << "  1. Exception: ArrcException -> SizeMismatch, NDimMismatch, etc.\n";
            cout << "  2. Function: Function -> Add, Mul, Sub, DivFunction\n";
            cout << "  3. Optimizer: Optimizer -> SGD, Adam, RMSProp\n";
            cout << "\nTemplate classes: NDArray<T>, Tensor<T>\n";
            cout << "Design Patterns: Strategy (Optimizer), Factory (functions::)\n";
            cout << "STL: vector, list, unordered_set, std::remove_if cu lambda\n";
            cout << "Memorie GPU: " << NDArray<float>::getTotalAllocatedMemory() << " bytes\n";
            break;
        }

        default:
            cout << "Optiune invalida!\n";
        }

        cout << "\nApasa Enter pentru a continua...";
        cin.ignore();
        cin.get();
    }

    return 0;
}
