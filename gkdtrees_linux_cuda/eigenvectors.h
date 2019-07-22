#ifndef EIGENVECTORS_H
#define EIGENVECTORS_H

#include <math.h>

class Eigenvectors {
  public:
    Eigenvectors(int in_dimensions, int out_dimensions) {
	d_in = in_dimensions;
	d_out = out_dimensions;
	
	covariance = new double[d_in*d_in];
	mean = new double[d_in];
	eigenvectors = new double[d_in*d_out];
	tmp = new double[d_in*d_out];
	computed = false;
	for (int i = 0; i < d_in; i++) {
	    mean[i] = 0;
	    for (int j = 0; j < d_in; j++) {
		covariance[i*d_in + j] = 0;		
		if (j < d_out) {
		    eigenvectors[j*d_in + i] = 0;
		    tmp[j*d_in + i] = 0;
		}
	    }
	}
	count = 0;
    }

    void add(const float *v) {
	for (int i = 0; i < d_in; i++) {
	    for (int j = 0; j < d_in; j++) {
		covariance[i*d_in+j] += v[i]*v[j];
	    }
	    mean[i] += v[i];
	}
	count++;
    }

    // how much of each eigenvector is in a particular vector?
    // multiply the vector by the transpose of the eigenvector matrix
    void apply(const float *v_in, float *v_out) {
	if (!computed) compute();
	
	for (int i = 0; i < d_out; i++) {
	    v_out[i] = 0;
	    for (int j = 0; j < d_in; j++) {
		v_out[i] += eigenvectors[i*d_in + j] * v_in[j];
	    }
	}
    }

    double *getEigenvector(int i) {
	if (!computed) compute();
	return eigenvectors + i*d_in;
    }

    void save(const char *filename) {
	if (!computed) compute();
	FILE *f = fopen(filename, "wb");
	fwrite(eigenvectors, sizeof(double), d_out*d_in, f);
	fclose(f);
    }

    void compute() {
	// first remove the mean and normalize by the count
	for (int i = 0; i < d_in; i++) {
	    for (int j = 0; j < d_in; j++) {
		covariance[i*d_in+j] -= mean[i]*mean[j]/count;
		covariance[i*d_in+j] /= count;
	    }
	}	

	// now compute the eigenvectors
	// TODO: do this using a non-retarded algorithm
	printf("Computing Eigenvectors\n");
	for (int i = 0; i < d_in; i++) {
	    for (int j = 0; j < d_out; j++) {
		eigenvectors[j*d_in + i] = covariance[i*d_in+j];
	    }
	}
	while (1) {
	    printf("Eigenvalues: ");
	    // orthonormalize
	    for (int i = 0; i < d_out; i++) {
		// first make this column independent of all the
		// previous columns		
		for (int j = 0; j < i; j++) {
		    // compute the dot product
		    double dot = 0;
		    for (int k = 0; k < d_in; k++) {
			dot += eigenvectors[i*d_in + k]*eigenvectors[j*d_in + k];
		    }
		    // The previous column is of unit length, so it's
		    // easy to make this one independent
		    for (int k = 0; k < d_in; k++) {
			eigenvectors[i*d_in + k] -= eigenvectors[j*d_in + k]*dot;
		    }
		}

		// now normalize this column
		double dot = 0;
		for (int k = 0; k < d_in; k++) {
		    dot += eigenvectors[i*d_in + k]*eigenvectors[i*d_in + k];
		}
		dot = sqrt(dot);

		printf("%3.3f ", dot);

		dot = 1.0/dot;

		// make sure the first element of each eigenvector is positive
		if (eigenvectors[i*d_in]*dot < 0) dot = -dot;

		for (int k = 0; k < d_in; k++) {
		    eigenvectors[i*d_in + k] *= dot;
		}

		dot = 0;
		for (int k = 0; k < d_in; k++) {
		    dot += eigenvectors[i*d_in + k]*eigenvectors[i*d_in + k];
		}
	    }
	    printf("\n");

	    /*
	    printf("eigenvector matrix:\n");
	    for (int i = 0; i < d_in; i++) {
		for (int j = 0; j < d_out; j++) {
		    printf("%3.4f ", eigenvectors[i*d_out+j]);
		}
		printf("\n");
	    }	    
	    */

	    // check for convergence
	    double dist = 0;
	    for (int j = 0; j < d_out; j++) {
		for (int i = 0; i < d_in; i++) {
		    double delta = tmp[j*d_in + i] - eigenvectors[j*d_in + i];
		    dist += delta*delta;
		}
	    }
	    printf("Distance to convergence: %f\n", dist);
	    if (dist < 0.001) break;
	    
	    // multiply by the covariance matrix
	    for (int i = 0; i < d_in; i++) {
		for (int j = 0; j < d_out; j++) {
		    tmp[j*d_in + i] = 0;
		    for (int k = 0; k < d_in; k++) {
			tmp[j*d_in + i] += covariance[i*d_in+k]*eigenvectors[j*d_in + k];
		    }
		}
	    }
	    double *t = tmp;
	    tmp = eigenvectors;
	    eigenvectors = t;

	    
	}


	computed = true;
    }

  private:
    int d_in, d_out;
    double *covariance, *mean, *eigenvectors, *tmp;
    bool computed;
    int count;
};

#endif
