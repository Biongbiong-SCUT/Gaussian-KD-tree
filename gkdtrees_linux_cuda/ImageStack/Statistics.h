#ifndef STATISTICS_H
#define STATISTICS_H

class Dimensions : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

class Stats {
  public:
    Stats(Window im);

    #define BASIC if (!basicStatsComputed) computeBasicStats();
    #define MOMENT if (!momentsComputed) computeMoments();

    inline double sum(int c)     {BASIC; return sums[c];}
    inline double sum()          {BASIC; return sum_;}
    inline double mean(int c)    {BASIC; return means[c];}
    inline double mean()         {BASIC; return mean_;}
    inline double minimum(int c) {BASIC; return mins[c];}
    inline double minimum()      {BASIC; return min_;}
    inline double maximum(int c) {BASIC; return maxs[c];}
    inline double maximum()      {BASIC; return max_;}
    inline int nans() {BASIC; return nans_;}
    inline int posinfs() {BASIC; return posinfs_;}
    inline int neginfs() {BASIC; return neginfs_;}
    inline double covariance(int c1, int c2) {MOMENT; return covarianceMatrix[c1 * channels + c2];}
    inline double variance(int c) {MOMENT; return variances[c];}
    inline double variance()      {MOMENT; return variance_;}
    inline double skew(int c)     {MOMENT; return skews[c];}
    inline double skew()          {MOMENT; return skew_;}
    inline double kurtosis(int c) {MOMENT; return kurtoses[c];}
    inline double kurtosis()      {MOMENT; return kurtosis_;}



    #undef BASIC
    #undef MOMENT

  private:
    void computeBasicStats();
    bool basicStatsComputed;    
    void computeMoments();
    bool momentsComputed;
    Window im_;

    int channels;
    vector<double> sums, means, variances, kurtoses, skews, mins, maxs;
    vector<double> covarianceMatrix;
    double sum_, mean_, variance_, min_, max_, kurtosis_, skew_;
    int nans_, neginfs_, posinfs_;
};

class Statistics : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im);
};

class Noise : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, float minVal, float maxVal);
};
 
class Histogram : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static vector< vector<float> > apply(Window im, int buckets = 256);    
}; 

class Equalize : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, float lower, float upper);
};

class HistogramMatch : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, Window model);
};

class Shuffle : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im);
};

class KMeans : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, int clusters);
};

class Sort : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, char dimension);
};

class DimensionReduction : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, int newChannels);
};

class LocalMaxima : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, bool tCheck, bool xCheck, bool yCheck, float threshold, string filename);
};

class Printf : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, string fmt, vector<float> args);
};

class FPrintf : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, string filename, string fmt, vector<float> args);    
};

#endif
