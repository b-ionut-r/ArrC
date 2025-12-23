//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef ARRC_SLICES_H
#define ARRC_SLICES_H

#include <vector>

class Slice {
    int start, stop, step;
public:
    Slice(int start, int stop, int step=1) : start(start), stop(stop), step(step) {};
    int size() const;
    void normalizeEnd(int shape_size);
    int getStart() const {return start;}
    int getStop() const {return stop;}
    int getStep() const {return step;}
    // Iterator
    class Iterator {
        int current;
        int stop;
        int step;
    public:
        Iterator(int current, int stop, int step) : current(current), stop(stop), step(step) {};
        // Dereference operator
        int operator*() const;
        // Pre-increment operator
        Iterator& operator++();
        // Post increment operator
        Iterator operator++(int);
        // Inequality operator
        bool operator!=(const Iterator &other) const;
        // Equality operator
        bool operator==(const Iterator &other) const;
    };
    Iterator begin() const;
    Iterator end() const;
};


#endif //ARRC_SLICES_H