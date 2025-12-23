//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#include "slices.h"
#include <vector>

int Slice::size() const {
    if (step > 0) {
        if (stop <= start) return 0;
        return (stop - start + step - 1) / step; // ceil
    } else if (step < 0) {
        if (stop >= start) return 0;
        return (start - stop - step - 1) / (-step); // ceil
    } else {
        return 0;
    }
}

void Slice::normalizeEnd(int shape_size) {
    if (stop < 0) {
        stop += shape_size;
    }
}

int Slice::Iterator::operator*() const {
    return current;
}

// Pre-increment operator
Slice::Iterator& Slice::Iterator::operator++() {
    current += step;
    return *this;
}

// Post increment operator
Slice::Iterator Slice::Iterator::operator++(int) {
    Iterator tmp = *this;
    current += step;
    return tmp;
}

// Inequality operator
bool Slice::Iterator::operator!=(const Iterator &other) const {
    // For range-based for loops: check if we've reached or passed the end
    // Also properly compare against other iterator for general use
    if (current == other.current) return false;
    if (step > 0) {
        return current < stop;
    } else {
        return current > stop;
    }
}

// Equality operator
bool Slice::Iterator::operator==(const Iterator &other) const {
    return current == other.current;
}

Slice::Iterator Slice::begin() const {
    return {start, stop, step};
}

Slice::Iterator Slice::end() const {
    return {stop, stop, step};
}