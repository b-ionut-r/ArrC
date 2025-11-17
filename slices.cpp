//
// Created by Bujor Ionut Raul on 16.11.2025.
//
#include <vector>


class Slice {
    int start, stop, step;
    int *indices;
public:
    Slice(int start, int stop, int step=1) : start(start), stop(stop), step(step) {};
    std::vector<int> getIndices() const {
        std::vector<int> indices;
        for (int idx: *this) {
            indices.push_back(idx);
        }
        return indices;
    }
    int size() const {
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
    void normalizeEnd(int shape_size) {
        if (stop < 0) {
            stop += shape_size;
        }
    }
    int getStart() const {
        return start;
    }
    int getStop() const {
        return stop;
    }
    int getStep() const {
        return step;
    }
    // Iterator
    class Iterator {
        int current;
        int stop;
        int step;
    public:
        Iterator(int current, int stop, int step) : current(current), stop(stop), step(step) {};
        // Dereference operator
        int operator*() const {
            return current;
        }
        // Pre-increment operator
        Iterator& operator++() {
            current += step;
            return *this;
        }
        // Post increment operator
        Iterator operator++(int) {
            Iterator tmp = *this;
            current += step;
            return tmp;
        }
        // Inequality operator
        bool operator!=(const Iterator &other) const {
            (void) other; // mark as unused
            if (step > 0) {
                return current < stop;
            }
            else {
                return current > stop;
            }
        }
        // Equality operator
        bool operator==(const Iterator &other) const {
            (void) other;
            return !(*this != other);
        }
    };
    Iterator begin() const {
        return Iterator(start, stop, step);
    }
    Iterator end() const {
        return Iterator(stop, stop, step);
    }
};
