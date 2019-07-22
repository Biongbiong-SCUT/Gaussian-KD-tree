#ifndef DUAL_H
#define DUAL_H

// A Dual stores a value, and the derivative of that value with respect to several other
// variables. 

#include <stdlib.h>
#include <math.h>

template<typename T>
class LazyArray {

    class ValRef {
        LazyArray<T> &owner;
        const int index;
      public:
        ValRef(LazyArray<T> &owner_, const int index_) : owner(owner_), index(index_) {}

        operator T() const {
            return owner.get(index);
        }

        void operator=(T v) {
            owner.set(index, v);
        }        

        void operator+=(T v) {
            owner.set(index, owner.get(index) + v);
        }

        void operator-=(T v) {
            owner.set(index, owner.get(index) - v);
        }

        void operator*=(T v) {
            owner.set(index, owner.get(index) * v);
        }

        void operator/=(T v) {
            owner.set(index, owner.get(index) / v);
        }
    };

    T *data;
    int *refCount;
    int minIndex, maxIndex;

    void incref() {
        if (data) refCount[0]++;
    };

    void decref() {
        if (data) {
            if (refCount[0] == 1) {
                free(refCount);
                free(data + minIndex);
		refCount = NULL;
		data = NULL;
            } else {
                refCount[0]--;
            }
        }
    };

  public:
    LazyArray() {
        refCount = NULL;
	data = NULL;
	minIndex = maxIndex = 0;
    }

    LazyArray(const LazyArray &other) {
	data = other.data;
        refCount = other.refCount;
        incref();
	minIndex = other.minIndex;
	maxIndex = other.maxIndex;
    }

    LazyArray &operator=(const LazyArray &other) {
	decref();
	data = other.data;
        refCount = other.refCount;
        incref();
	minIndex = other.minIndex;
	maxIndex = other.maxIndex;
	return *this;
    }    

    ~LazyArray() {
        decref();
    }

    void resize(int newMin, int newMax) {
	if (newMin < minIndex || newMax > maxIndex || data == NULL || refCount[0] > 1) {
	    T *oldData = data;
            int *oldRef = refCount;
	    if (oldData) {
		if (minIndex < newMin) newMin = minIndex;
		if (maxIndex > newMax) newMax = maxIndex;
	    }
	    data = (T *)malloc((newMax - newMin)*sizeof(T));
            refCount = (int *)malloc(sizeof(int));
	    data -= newMin;
	    if (oldData) {
                for (int i = newMin; i < minIndex; i++) {
                    data[i] = 0;
                }
		for (int i = minIndex; i < maxIndex; i++) {
		    data[i] = oldData[i];
		}
		for (int i = maxIndex; i < newMax; i++) {
		    data[i] = 0;
		}
		if (oldRef[0] == 1) {
                    free(oldData + minIndex);
                    free(oldRef);
                } else {
                    oldRef[0]--;
                }
	    } else {
                for (int i = newMin; i < newMax; i++) {
                    data[i] = 0;
                }
            }
	    minIndex = newMin;
	    maxIndex = newMax;
            refCount[0] = 1;
	}
    }

    void resize(int idx) {
	resize(idx, idx+1);
    }

    T get(int idx) const {
	if (data && idx >= minIndex && idx < maxIndex) return data[idx];
	else return 0;        
    }

    void set(int idx, T val) {
        //assert(val < 1000 && val > -1000, "Setting to very large value %f\n", val);
        resize(idx);
        data[idx] = val;
    }

    ValRef operator[](int idx) {
        return ValRef(*this, idx);
    }

    T operator[](int idx) const {
        return get(idx);
    }

    int lowerBound() const {return minIndex;} 
    int upperBound() const {return maxIndex;}
    bool isEmpty() const {return data == NULL;}

/*
    void debug() const {
        printf("LazyArray at %lx \n", this);
        for (int i = lowerBound(); i < upperBound(); i++) {
            printf("%i: %f\n", i, data[i]);
        }
    }
*/
};

class Dual {
  public:
    LazyArray<float> derivative;
    float value;

    Dual() { // initializes to the constant zero	
	value = 0;
    }

    Dual(float val) { // initializes to a constant 
	value = val;
    }

    Dual(float val, int d) { // initializes to a variable with given value
	value = val;
	derivative[d] = 1;
    }

    bool isConstant() const {return derivative.isEmpty();}

/*
    void debug() {
	printf("Dual at %llx = %f\n", (unsigned long long)this, value);
	if (isConstant()) printf("Constant\n");
	else {
	    printf("Derivatives: \n");
	    for (int i = derivative.lowerBound(); i < derivative.upperBound(); i++) {
		printf("%i: %f\n", i, (float)derivative[i]);
	    }
	}
	printf("\n");
    }
*/

    void operator+=(const Dual &b) {
	value += b.value;

	if (b.isConstant()) return; 
	derivative.resize(b.derivative.lowerBound(), b.derivative.upperBound());

	for (int d = b.derivative.lowerBound(); d < b.derivative.upperBound(); d++) {
	    derivative[d] += b.derivative[d];
	}
    }

    void operator*=(const Dual &b) {
	if (b.isConstant() && !isConstant()) {
	    for (int d = derivative.lowerBound(); d < derivative.upperBound(); d++) {
		derivative[d] *= b.value;
	    }
	} else {
	    derivative.resize(b.derivative.lowerBound(), b.derivative.upperBound());
	    
	    for (int d = derivative.lowerBound(); d < derivative.upperBound(); d++) {
		derivative[d] = (derivative[d] * b.value + value * b.derivative[d]);
	    }
	}
	value *= b.value;
    }

    void operator-=(const Dual &b) {
	value -= b.value;

	if (b.isConstant()) return; 
	derivative.resize(b.derivative.lowerBound(), b.derivative.upperBound());

	for (int d = b.derivative.lowerBound(); d < b.derivative.upperBound(); d++) {
	    derivative[d] -= b.derivative[d];
	}
    }

    void operator/=(const Dual &b) {
	float mult = 1.0f / b.value;
	value *= mult;

	if (b.isConstant() && !isConstant()) {
	    for (int d = derivative.lowerBound(); d < derivative.upperBound(); d++) {
		derivative[d] *= mult;
	    }	    
	} else {
	    derivative.resize(b.derivative.lowerBound(), b.derivative.upperBound());
	    
	    for (int d = derivative.lowerBound(); d < derivative.upperBound(); d++) {
		derivative[d] = (derivative[d] - value * b.derivative[d]) * mult;
	    }
	}
    }

};

inline Dual operator+(const Dual &a, const Dual &b) {
    Dual result(a);
    result += b;
    return result;
}

inline Dual operator*(const Dual &a, const Dual &b) {
    Dual result(a);
    result *= b;
    return result;
}

inline Dual operator-(const Dual &a, const Dual &b) {
    Dual result(a);
    result -= b;
    return result;
}

inline Dual operator/(const Dual &a, const Dual &b) {
    Dual result(a);
    result /= b;
    return result;
}

// some useful functions

inline float sqrt(float x) {return ::sqrtf(x);}
inline float sin(float x) {return ::sinf(x);}
inline float cos(float x) {return ::cosf(x);}
inline float pow(float a, float b) {return ::powf(a, b);}

inline Dual sqrt(const Dual &a) {
    Dual result(a);
    result.value = sqrtf(result.value);
    if (a.isConstant()) return result;

    float mult = 0.5f / result.value;
    for (int i = a.derivative.lowerBound(); i < a.derivative.upperBound(); i++) {
	result.derivative[i] *= mult;
    }
    return result;
}


inline Dual pow(const Dual &a, float b) {
    Dual result(a);
    float mult = powf(a.value, b-1);
    result.value *= mult;
    if (a.isConstant()) return result;
    mult *= b;
    for (int i = a.derivative.lowerBound(); i < a.derivative.upperBound(); i++) {
	result.derivative[i] *= mult;
    }
    return result;
}

inline Dual pow(float a, const Dual &b) {
    Dual result(b);
    result.value = powf(a, b.value);
    if (b.isConstant()) return result;
    float mult = result.value * logf(a);
    for (int i = b.derivative.lowerBound(); i < b.derivative.upperBound(); i++) {
        result.derivative[i] *= mult;
    }
}


inline Dual exp(const Dual &a) {
    Dual result(a);
    result.value = expf(result.value);
    if (a.isConstant()) return result;

    for (int i = a.derivative.lowerBound(); i < a.derivative.upperBound(); i++) {
	result.derivative[i] *= result.value;
    }
    return result;
}

inline float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

inline Dual sigmoid(const Dual &a) {
    Dual result(a);
    result.value = sigmoid(result.value);
    if (a.isConstant()) return result;

    float mult = result.value * (1 - result.value);
    for (int i = a.derivative.lowerBound(); i < a.derivative.upperBound(); i++) {
	result.derivative[i] *= mult;
    }
    return result;
}

inline Dual sin(const Dual &a) {
    Dual result(a);
    result.value = sin(result.value);
    if (a.isConstant()) return result;
    float mult = cos(a.value);
    for (int i = a.derivative.lowerBound(); i < a.derivative.upperBound(); i++) {
	result.derivative[i] *= mult;
    }
    return result;
}


inline Dual cos(const Dual &a) {
    Dual result(a);
    result.value = cos(result.value);
    if (a.isConstant()) return result;
    float mult = -sin(a.value);
    for (int i = a.derivative.lowerBound(); i < a.derivative.upperBound(); i++) {
	result.derivative[i] *= mult;
    }
    return result;
}

// testing code
/*


void gradientDescent(Dual (*errorfunc)(float *state), float *initialState, float stepsize, int d) {
    // apply the error function to the state
    Dual error = (*errorfunc)(initialState);
	
    // move the state according to the gradients
    for (int k = 0; k < d; k++) {
	initialState[k] += stepsize * error.derivative[k];
    }
    stepsize *= stepsizeReduction;
}

// uses newton's method
void findRoot(Dual (*func)(float *input), float *state, int iterations, int d) {
    float stepsize = -1.0f / d;

    for (int i = 0; i < iterations; i++) {
	// apply the function to the state
	Dual value = (*func)(state);

	// move the each state variable enough to zero the function
	for (int k = 0; k < d; k++) {
	    state[k] += stepsize * value.value / value.derivative[k];
	}
    }
}



#include <stdio.h>
#include <stdlib.h>

Dual sampleErrorFunc(float *state) {
    Dual x(state[0], 0);
    Dual y(state[1], 1); 
    Dual error = (x - 5) * (x - 5);
    error += (y + 5) * (y + 5);
    error -= 3;
    return error;
}

int main(int argc, const char **argv) {

    float state[2];

    printf("\n\nAttempting to zero the function with Newton's method\n"); 
    state[0] = 23; 
    state[1] = -35; 
    for (int i = 0; i < 30; i++) {
	printf("f(%f, %f) = %f\n", state[0], state[1], sampleErrorFunc(state).value);
	findRoot(sampleErrorFunc, state, 1, 2);
    }
    printf("f(%f, %f) = %f\n", state[0], state[1], sampleErrorFunc(state).value);

    printf("\n\nFinding the minimum of the function with gradient descent (should be 5, 5)\n");
    state[0] = 23; 
    state[1] = -35; 
    float ss = 0.1;
    for (int i = 0; i < 30; i++) {
	printf("f(%f, %f) = %f\n", state[0], state[1], sampleErrorFunc(state).value);
	gradientDescent(sampleErrorFunc, state, 1, ss, 10, 2);
    }
    printf("f(%f, %f) = %f\n", state[0], state[1], sampleErrorFunc(state).value);

    return 0;
}

*/



#endif
