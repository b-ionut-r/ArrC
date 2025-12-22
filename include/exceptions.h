//
// Created by Bujor Ionut Raul on 16.12.2025.
//

#ifndef ARRC_EXCEPTIONS_H
#define ARRC_EXCEPTIONS_H

#include <exception>
#include <string>
using namespace std;

class ArrcException : public exception {
protected:
    string message;
public:
    explicit ArrcException(const string &msg) : message("ArrC Error: " + msg) {}
    const char* what() const noexcept override {
        return message.c_str();
    }
};

class SizeMismatchException : public ArrcException {
public:
    SizeMismatchException(const string &msg)
        : ArrcException("Size Mismatch -> " + msg) {}
};

class NDimMismatchException : public ArrcException {
public:
    NDimMismatchException(const string &msg)
        : ArrcException("NDim Mismatch -> " + msg) {}
};


class ShapeMismatchException : public ArrcException {
public:
    ShapeMismatchException(const string &msg)
        : ArrcException("Shape Mismatch -> " + msg) {}
};


class IndexingException : public ArrcException {
public:
    IndexingException(const string &msg)
        : ArrcException("Indexing/Slicing Error -> " + msg) {}
};

class CudaKernelException : public ArrcException {
public:
    explicit CudaKernelException(const string &cudaError)
        : ArrcException("CUDA Kernel Failure -> " + cudaError) {}
};

#endif // ARRC_EXCEPTIONS_H

