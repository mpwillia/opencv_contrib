/*
 * Copyright (c) 2011,2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */
#include "precomp.hpp"
#include "opencv2/ddface/ddfacerec.hpp"
#include "face_basic.hpp"
#include "mcl.hpp"
#include "cluster.hpp"
#include "modelstorage.hpp"

#include "tbb/tbb.h"
#include "tbb/task_scheduler_init.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <thread>

#define COMP_ALG HISTCMP_CHISQR_ALT
//#define COMP_ALG HISTCMP_BHATTACHARYYA
#define SIZEOF_CV_32FC1 4

namespace cv { namespace face {

// Face Recognition based on Local Binary Patterns.
//
//  Ahonen T, Hadid A. and Pietikäinen M. "Face description with local binary
//  patterns: Application to face recognition." IEEE Transactions on Pattern
//  Analysis and Machine Intelligence, 28(12):2037-2041.
//
class xLBPH : public xLBPHFaceRecognizer
{
private:
    // alg settings
    int _grid_x;
    int _grid_y;
    int _radius;
    int _neighbors;
    double _threshold;
    
    // model path info
    String _modelpath;
    String _modelname;

    ModelStorage _model;            

    // label info
    std::map<int, int> _labelinfo;

    // histograms
    std::map<int, std::vector<Mat>> _histograms;
    std::map<int, Mat> _histavgs;
    std::map<int, Mat> _distmats;    
    std::map<int, std::vector<cluster::cluster_t>> _clusters;

    // defines what prediction algorithm to use
    int _algToUse;
    
    bool _useClusters;

    //--------------------------------------------------------------------------
    // Multithreading
    // REMOVE: we're usig TBB now don't need most of this
    //--------------------------------------------------------------------------
    // REMOVE: dont need custom dispatchers
    template <typename S, typename D>
    void performMultithreadedCalc(const std::vector<S> &src, std::vector<D> &dst, int numThreads, void (xLBPH::*calcFunc)(const std::vector<S> &src, std::vector<D> &dst) const) const;
    template <typename Q, typename S, typename D>
    void performMultithreadedComp(const Q &query, const std::vector<S> &src, std::vector<D> &dst, int numThreads, void (xLBPH::*compFunc)(const Q &query, const std::vector<S> &src, std::vector<D> &dst) const) const;
    
    // still useful for setting max TBB threads
    int _maxThreads;
    
    //--------------------------------------------------------------------------
    // Model Training Function
    //--------------------------------------------------------------------------
    // Computes a LBPH model with images in src and
    // corresponding labels in labels, possibly preserving
    // old model data.
    void train(InputArrayOfArrays src, InputArray labels, bool preserveData);

    void calculateLabels(const std::vector<std::pair<int, std::vector<Mat>>> &labelImages, std::vector<std::pair<int, int>> &labelinfo) const;
    void calculateHistograms(const std::vector<Mat> &src, std::vector<Mat> &dst) const;

    //--------------------------------------------------------------------------
    // Prediction Functions
    //--------------------------------------------------------------------------
    void predict_std(InputArray _src, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::vector<int> &labels) const;
    void predict_avg(InputArray _src, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::vector<int> &labels) const;
    void predict_avg_clustering(InputArray _src, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::vector<int> &labels) const;

    void compareLabelHistograms(const Mat &query, const std::vector<std::pair<int, std::vector<Mat>>> &labelhists, std::vector<std::pair<int, std::vector<double>>> &labeldists) const;
    void compareHistograms(const Mat &query, const std::vector<Mat> &hists, std::vector<double> &dists) const;


    int minLabelsToCheck = 10;
    double labelsToCheckRatio = 0.05;

    int minClustersToCheck = 3;
    double clustersToCheckRatio = 0.20;


    //--------------------------------------------------------------------------
    // Managing Histogram Binary Files 
    // REMOVE: None of this sshould really be needed anymore 
    //         it should all be abstracted away with ModelStorage
    //--------------------------------------------------------------------------
    // Reading/Writing Histogram Files
    bool readHistograms(const String &filename, std::vector<Mat> &histograms) const;
    bool writeHistograms(const String &filename, const std::vector<Mat> &histograms, bool appendhists) const;
    
    // Saving/Loading/Updating Histogram File by Label
    bool saveHistograms(int label, const std::vector<Mat> &histograms) const;
    bool updateHistograms(int label, const std::vector<Mat> &histrograms) const;
    bool loadHistograms(int label, std::vector<Mat> &histograms) const;
   
    // Memory mapping histograms
    void mmapHistograms();
    void munmapHistograms();

    
    //--------------------------------------------------------------------------
    // Data Management Strategy/Technique Functions
    //--------------------------------------------------------------------------
    // Histogram Averages
    void calcHistogramAverages(std::vector<Mat> &histavgs) const;

    // REMOVE: Multithreading handled by TBB now
    void calcHistogramAverages_thread(const std::vector<int> &labels, std::vector<Mat> &avgsdst) const;

    // REMOVE: mmaping and loading should now be handled by ModelStorage
    bool loadHistogramAverages(std::map<int, Mat> &histavgs) const;
    void mmapHistogramAverages();

    //--------------------------------------------------------------------------
    // Histogram Clustering and Markov Clustering
    // REMOVE: All of the clustering should now be in cluster.hpp and mcl.hpp
    //         where cluster.hpp is the main entry/exit point for clustering
    //         and mcl.hpp is the core MCL clustering algorithm. 
    //         cluster.hpp wraps mcl.hpp and hsould in theory be able to support
    //         aditional clustering algorithms without changing the entry/exit headers.
    //--------------------------------------------------------------------------
    void clusterHistograms(std::map<int, std::vector<cluster::cluster_t>> &clusters) const;
    void cluster_calc_weights(Mat &dists, Mat &weights, double tierStep, int numTiers);
    void cluster_dists(Mat &dists, Mat &mclmat, double r);
    void cluster_interpret(Mat &mclmat, std::vector<std::set<int>> &clusters);
    double cluster_ratio(std::vector<std::set<int>> &clusters);
    void cluster_find_optimal(Mat &dists, std::vector<std::set<int>> &clusters);
    
    void cluster_label(int label, std::vector<std::pair<Mat, std::vector<Mat>>> &matClusters);
    //void cluster_label(int label, std::vector<std::set<int>> &clusters);

    //void cluster_label(int label, std::vector<std::pair<Mat, std::vector<Mat>>> &clusters);
        
    // NOTE: This could still be useful
    void printMat(const Mat &mat, int label) const;
    
    // NOTE: Both the clustering ahd MCL settings are still used
    // IDEA: Perhaps we can abstract these both away in cluster.hpp and mcl.hpp?
    // Histogram Clustering - Settings
    double cluster_tierStep = 0.01; // Sets how large a tier is, default is 0.01 or 1%
    int cluster_numTiers = 10; // Sets how many tiers to keep, default is 10, or 10% max tier
    int cluster_max_iterations = 5;

    // Markov Clustering Algorithm (MCL) - Settings
    /* Sets the number of MCL iterations, default is 10
     * If 0 then iterates until no change is found
     */
    int mcl_iterations = 10;
    int mcl_expansion_power = 2; // Sets the expansion power exponent, default is 2
    double mcl_inflation_power = 2; // Sets the inflation power exponent, default is 2 
    double mcl_prune_min = 0.001; // Sets the minimum value to prune, any values below this are set to zero, default is 0.001

    //--------------------------------------------------------------------------
    // Misc 
    // ???: How much of this is still needed?
    //--------------------------------------------------------------------------
    bool exists(const String &filename) const;
    int getHistogramSize() const;
    bool matsEqual(const Mat &a, const Mat &b) const;
    void averageHistograms(const std::vector<Mat> &hists, Mat &histavg) const;


public:
    using FaceRecognizer::save;
    //using FaceRecognizer::load;

    // Initializes this xLBPH Model. The current implementation is rather fixed
    // as it uses the Extended Local Binary Patterns per default.
    //
    // radius, neighbors are used in the local binary patterns creation.
    // grid_x, grid_y control the grid size of the spatial histograms.
    xLBPH(int radius_=1, int neighbors_=8,
            int gridx=8, int gridy=8,
            double threshold = DBL_MAX,
            String modelpath="") :
                _grid_x(gridx),
                _grid_y(gridy),
                _radius(radius_),
                _neighbors(neighbors_),
                _threshold(threshold),
                _model(modelpath, radius_, neighbors_, gridx, gridy) {

        _algToUse = 0;
        _useClusters = true;
        _maxThreads = tbb::task_scheduler_init::automatic;
        setModelPath(modelpath);
    }

    // Initializes and computes this xLBPH Model. The current implementation is
    // rather fixed as it uses the Extended Local Binary Patterns per default.
    //
    // (radius=1), (neighbors=8) are used in the local binary patterns creation.
    // (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
    xLBPH(InputArrayOfArrays src,
            InputArray labels,
            int radius_=1, int neighbors_=8,
            int gridx=8, int gridy=8,
            double threshold = DBL_MAX,
            String modelpath="") :
                _grid_x(gridx),
                _grid_y(gridy),
                _radius(radius_),
                _neighbors(neighbors_),
                _threshold(threshold),
                _model(modelpath, radius_, neighbors_, gridx, gridy) {
        _algToUse = 0;
        _useClusters = true;
        _maxThreads = tbb::task_scheduler_init::automatic;
        setModelPath(modelpath);
        train(src, labels);
    }

    ~xLBPH() { }

    //TODO: Clean all of this up and add more subheadings to make it easier to read

    // Computes a xLBPH model with images in src and
    // corresponding labels in labels.
    void train(InputArrayOfArrays src, InputArray labels);

    // Updates this xLBPH model with images in src and
    // corresponding labels in labels.
    void update(InputArrayOfArrays src, InputArray labels);

    // Predicts the label of a query image in src.
    int predict(InputArray src) const;

    // Predicts the label and confidence for a given sample.
    void predict(InputArray _src, int &label, double &dist) const;

    // predictMulti
    void predictMulti(InputArray _src, OutputArray _preds, int numPreds) const;
    void predictMulti(InputArray _src, OutputArray _preds, int numPreds, InputArray _labels) const;
    void predictAll(std::vector<Mat> &_src, std::vector<Mat> &_preds, int numPreds, InputArray _labels) const;

    // See FaceRecognizer::load.
    void load(const FileStorage& fs);
    void load(const String &filename);
    bool load();

    // See FaceRecognizer::save.
    void save(FileStorage& fs) const;
    
    CV_IMPL_PROPERTY(int, GridX, _grid_x)
    CV_IMPL_PROPERTY(int, GridY, _grid_y)
    CV_IMPL_PROPERTY(int, Radius, _radius)
    CV_IMPL_PROPERTY(int, Neighbors, _neighbors)
    CV_IMPL_PROPERTY(double, Threshold, _threshold)
        
    // path getters/setters
    // REMOVE: All of these should be abstracted away by ModelStorage and there
    //         is no reason an external class should be dipping into the model
    void setModelPath(String modelpath);
    String getModelPath() const;
    String getModelName() const;
    String getInfoFile() const;
    String getHistogramsDir() const;
    String getHistogramFile(int label) const;
    String getHistogramAveragesFile() const;


    //--------------------------------------------------------------------------
    // Additional Public Functions 
    // NOTE: Remember to add header to opencv2/face/facerec.hpp
    //--------------------------------------------------------------------------
    
    //Prediction Algorithm Setters
    void setAlgToUse(int alg);
    void setLabelsToCheck(int min, double ratio);
    void setClustersToCheck(int min, double ratio);
    void setMCLSettings(int numIters, int e, double r);
    void setClusterSettings(double tierStep, int numTiers, int maxIters);
    
    //Prediction Algorithm Getters
    int getAlgUsed() const;
    int getLabelsToCheckMin() const;
    double getLabelsToCheckRatio() const;
    int getClustersToCheckMin() const;
    double getClustersToCheckRatio() const;
    int getMCLIters() const;
    int getMCLExpansionPower() const;
    double getMCLInflationPower() const;
    double getClusterTierStep() const;
    int getClusterNumTiers() const;
    int getClusterMaxIters() const;

    // ???: do we need this anymore?
    void setUseClusters(bool flag);

    // Long awaited getters
    // Broad Information Getters
    void getLabelInfo(OutputArray labelinfo) const;
    int getNumLabels() const;
    int getTotalHists() const;

    // Label Specific Information Getters
    bool isTrainedFor(int label) const;
    int getNumHists(int label) const;
    int getNumClusters(int label) const;
        
    // Threading Setters/Getters
    void setMaxThreads(int max);
    int getMaxThreads() const;


    void test();

};

// Threading Setters/Geters
void xLBPH::setMaxThreads(int max) {
    if(max == 0)
        _maxThreads = 1;
    else if(max < 0)
        _maxThreads = tbb::task_scheduler_init::automatic;
    else if(max > tbb::task_scheduler_init::default_num_threads())
        _maxThreads = tbb::task_scheduler_init::default_num_threads();
    else
        _maxThreads = max;
} 

int xLBPH::getMaxThreads() const {
    return _maxThreads; 
} 


// Broad Info Getters
void xLBPH::getLabelInfo(OutputArray output) const {
    
    output.create((int)_labelinfo.size(), 2, CV_32SC1);
    Mat outmat = output.getMat();
    
    int i = 0;
    for(std::map<int,int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); it++) {
        outmat.at<int>(i, 0) = it->first;
        outmat.at<int>(i, 1) = it->second;
        i++;
    } 
        
} 

int xLBPH::getNumLabels() const {
    return (int)_labelinfo.size();
} 

int xLBPH::getTotalHists() const {
    int sum = 0;
    for(std::map<int,int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); it++)
        sum += it->second; 
    return sum;
} 

// Label Specific Info Getters
bool xLBPH::isTrainedFor(int label) const {
    return _labelinfo.find(label) != _labelinfo.end();
} 

int xLBPH::getNumHists(int label) const {
    if(isTrainedFor(label))
        return _labelinfo.at(label);
    else
        return -1;
} 

int xLBPH::getNumClusters(int label) const {
    if(isTrainedFor(label))
        return (int)_clusters.at(label).size();
    else
        return -1;
} 

// predicion Algorithm Getters



// Prediction Algorithm Setters/Getters
void xLBPH::setUseClusters(bool flag) {
    _useClusters = flag;
}

void xLBPH::setClusterSettings(double tierStep, int numTiers, int maxIters) {
    cluster_tierStep = tierStep;
    cluster_numTiers = numTiers;
    cluster_max_iterations = maxIters;
} 
double xLBPH::getClusterTierStep() const {return cluster_tierStep;}
int xLBPH::getClusterNumTiers() const {return cluster_numTiers;}
int xLBPH::getClusterMaxIters() const {return cluster_max_iterations;}

void xLBPH::setMCLSettings(int numIters, int e, double r) {
    mcl_iterations = numIters;
    mcl_expansion_power = e;
    mcl_inflation_power = r;
}
int xLBPH::getMCLIters() const {return mcl_iterations;}
int xLBPH::getMCLExpansionPower() const {return mcl_expansion_power;}
double xLBPH::getMCLInflationPower() const {return mcl_inflation_power;}

void xLBPH::setLabelsToCheck(int min, double ratio) {
    minLabelsToCheck = min;
    labelsToCheckRatio = ratio;
} 
int xLBPH::getLabelsToCheckMin() const {return minLabelsToCheck;} 
double xLBPH::getLabelsToCheckRatio() const {return labelsToCheckRatio;} 

void xLBPH::setClustersToCheck(int min, double ratio) {
    minClustersToCheck = min;
    clustersToCheckRatio = ratio;
} 
int xLBPH::getClustersToCheckMin() const {return minClustersToCheck;} 
double xLBPH::getClustersToCheckRatio() const {return clustersToCheckRatio;} 


void xLBPH::setAlgToUse(int alg) {
    _algToUse = alg; 
}

int xLBPH::getAlgUsed() const {
    return _algToUse;
} 

//------------------------------------------------------------------------------
// Model Path and Model File Getters/Setters 
//------------------------------------------------------------------------------

// Sets _modelpath, extracts model name from path, and sets _modelname

// TODO: Replace with modelstorage
void xLBPH::setModelPath(String modelpath) {
    
    // given path can't be empty
    CV_Assert(modelpath.length() > 0);

    // path can't contain "//"
    CV_Assert((int)modelpath.find("//") == -1);
    
    // find last index of '/' 
    size_t idx = modelpath.find_last_of('/');

    if((int)idx < 0) {
        _modelpath = modelpath;
        _modelname = modelpath;
    }
    else if((int)idx >= (int)modelpath.length()-1) {
        setModelPath(modelpath.substr(0, modelpath.length()-1));
    }
    else {
        _modelpath = modelpath;
        _modelname = _modelpath.substr(idx + 1);
    }

}

// TODO: Replace with modelstorage
String xLBPH::getModelPath() const {
    return _modelpath; 
}

// TODO: Replace with modelstorage
String xLBPH::getModelName() const {
    return _modelname;
} 

// TODO: Replace with modelstorage
String xLBPH::getInfoFile() const {
    return getModelPath() + "/" + getModelName() + ".yml";
}

// TODO: Replace with modelstorage
String xLBPH::getHistogramsDir() const {
    return getModelPath() + "/" + getModelName() + "-histograms";
}

// TODO: Replace with modelstorage
String xLBPH::getHistogramFile(int label) const {
    char labelstr[16];
    sprintf(labelstr, "%d", label);
    return getHistogramsDir() + "/" + getModelName() + "-" + labelstr + ".bin";
}

// TODO: Replace with modelstorage
String xLBPH::getHistogramAveragesFile() const {
    return getHistogramsDir() + "/" + getModelName() + "-averages.bin";
}

//------------------------------------------------------------------------------
// Additional Functions and File IO
//------------------------------------------------------------------------------
static String matToHex(const Mat &mat) {
    String s = "";
    
    const unsigned char* data = mat.ptr<unsigned char>();

    for(int i = 0; i < mat.cols; i++) {
        char valuestr[32];
        int idx = i*4;
        sprintf(valuestr, "%02x%02x ", *(data+idx), *(data+idx+1));
        s += valuestr;
        sprintf(valuestr, "%02x%02x ", *(data+idx+2), *(data+idx+3));
        s += valuestr;
    }
    return s;
}

static String matToString(const Mat &mat) {
    String s = "[";

    for(int i = 0; i < mat.cols; i++) {
        if(i != 0)
            s += ", ";

        char valuestr[64];
        sprintf(valuestr, "%f", mat.at<float>(i));
        s += valuestr;
    }
    s += "]";
    
    return s;
}

bool checkBool(bool expected, bool got) {
    bool result = (expected == got);
    printf("%s - Expects %s : Got %s\n", (result ? "PASS" : "FAIL"), (expected ? "true" : "false"), (got ? "true" : "false"));
    return result;
}

bool checkStr(String expected, String got) {
    bool result = (expected.compare(got) == 0);
    printf("%s - Expects \"%s\" : Got \"%s\"\n", (result ? "PASS" : "FAIL"), expected.c_str(), got.c_str());
    return result;
} 

void xLBPH::test() {

    printf(" -=### Running xLBPH Tests ###=- \n");
    printf(" - Testing TBB Max Threads Control\n");
    const int numValues = 10000000;
    printf(" - Setting up test...\n", numValues);
    tbb::concurrent_vector<int> values;
    for(int i = 0; i < numValues; i++) {
        values.push_back(i);
    }
    
    printf(" - Starting test...\n", numValues);
    int n = tbb::task_scheduler_init::default_num_threads();
    printf("Default Num Threads: %d\n", n);

    for( int p=1; p<=n; ++p ) {
        printf("\rTesting with %d threads...", p);
        // Construct task scheduler with p threads
        tbb::task_scheduler_init init(p);
        tbb::tick_count t0 = tbb::tick_count::now();
        
        tbb::atomic<long> globalSum;
        tbb::parallel_for_each(values.begin(), values.end(),
            [&](int val) {
                long sum = 0;
                while(val > 0) {
                    sum++;
                    val--;
                }

                globalSum.fetch_and_add(sum);
            }
        );

        tbb::tick_count t1 = tbb::tick_count::now();
        double t = (t1-t0).seconds();
        printf("With %d threads time = %.3f   \n", p, t);
         // Implicitly destroy task scheduler.
    }

}


bool xLBPH::matsEqual(const Mat &a, const Mat &b) const {
    return countNonZero(a!=b) == 0; 
}

int xLBPH::getHistogramSize() const {
    return (int)(std::pow(2.0, static_cast<double>(_neighbors)) * _grid_x * _grid_y);
}

// TODO: Replace with modelstorage
bool xLBPH::exists(const String &filepath) const {
    struct stat buffer;   
    return (stat (filepath.c_str(), &buffer) == 0);   
}


// TODO: Replace with modelstorage
// Wrapper functions for load/save/updating histograms for specific labels
bool xLBPH::loadHistograms(int label, std::vector<Mat> &histograms) const {
    return readHistograms(getHistogramFile(label), histograms);
}

bool xLBPH::saveHistograms(int label, const std::vector<Mat> &histograms) const {
    return writeHistograms(getHistogramFile(label), histograms, false);
}

bool xLBPH::updateHistograms(int label, const std::vector<Mat> &histograms) const {
    return writeHistograms(getHistogramFile(label), histograms, true);
}


// TODO: Replace with modelstorage
// Main read/write functions for histograms
bool xLBPH::readHistograms(const String &filename, std::vector<Mat> &histograms) const {
    FILE *fp = fopen(filename.c_str(), "r");
    if(fp == NULL) {
        //std::cout << "cannot open file at '" << filename << "'\n";
        return false;
    }
    
    float buffer[getHistogramSize()];
    while(fread(buffer, sizeof(float), getHistogramSize(), fp) > 0) {
        Mat hist = Mat::zeros(1, getHistogramSize(), CV_32FC1);
        memcpy(hist.ptr<float>(), buffer, getHistogramSize() * sizeof(float));
        histograms.push_back(hist);
    }
    fclose(fp);
    return true;
}


// TODO: Replace with modelstorage
bool xLBPH::writeHistograms(const String &filename, const std::vector<Mat> &histograms, bool appendhists) const {
    FILE *fp = fopen(filename.c_str(), (appendhists == true ? "a" : "w"));
    if(fp == NULL) {
        //std::cout << "cannot open file at '" << filename << "'\n";
        return false;
    }

    float* buffer = new float[getHistogramSize() * (int)histograms.size()];
    for(size_t sampleIdx = 0; sampleIdx < histograms.size(); sampleIdx++) {
        float* writeptr = buffer + ((int)sampleIdx * getHistogramSize());
        memcpy(writeptr, histograms.at((int)sampleIdx).ptr<float>(), getHistogramSize() * sizeof(float));
    }
    fwrite(buffer, sizeof(float), getHistogramSize() * (int)histograms.size(), fp);
    delete buffer;

    fclose(fp);
    return true;
}

void xLBPH::averageHistograms(const std::vector<Mat> &hists, Mat &histavg) const {
    histavg = Mat::zeros(1, getHistogramSize(), CV_64FC1);

    for(size_t idx = 0; idx < hists.size(); idx++) {
        Mat dst;
        hists.at((int)idx).convertTo(dst, CV_64FC1);
        histavg += dst; 
    }
    histavg /= (int)hists.size();
    histavg.convertTo(histavg, CV_32FC1);
}

// REMOVE: Don't need since we started using TBB
void xLBPH::calcHistogramAverages_thread(const std::vector<int> &labels, std::vector<Mat> &avgsdst) const {
    for(size_t idx = 0; idx < labels.size(); idx++) {
        Mat histavg;
        averageHistograms(_histograms.at(labels.at(idx)), histavg);
        avgsdst.push_back(histavg);
    } 
}

void xLBPH::calcHistogramAverages(std::vector<Mat> &histavgs) const {
    
    int histsize = getHistogramSize();

    tbb::concurrent_vector<std::pair<int, Mat>> concurrent_averages;

    tbb::parallel_for_each(_histograms.begin(), _histograms.end(),
        [&concurrent_averages, histsize](std::pair<int, std::vector<Mat>> it) {
            //Mat histavg;
            //this->averageHistograms(it.second, histavg);
            
            std::vector<Mat> hists = it.second;
            Mat histavg = Mat::zeros(1, histsize, CV_64FC1);

            for(size_t idx = 0; idx < hists.size(); idx++) {
                Mat dst;
                hists.at((int)idx).convertTo(dst, CV_64FC1);
                histavg += dst; 
            }
            histavg /= (int)hists.size();
            histavg.convertTo(histavg, CV_32FC1);

            concurrent_averages.push_back(std::pair<int, Mat>(it.first, histavg));
        } 
    );

    //std::vector<Mat> averages;
    for(std::map<int,int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); it++) {
        int label = it->first; 

        //find it in concurrent_averages
        for(tbb::concurrent_vector<std::pair<int, Mat>>::const_iterator avg = concurrent_averages.begin(); avg != concurrent_averages.end(); avg++) {
            if(avg->first == label) {
                histavgs.push_back(avg->second);
                break;
            }
        }
    }
        
    //return writeHistograms(getHistogramAveragesFile(), averages, false);
}

// TODO: Replace with modelstorage
bool xLBPH::loadHistogramAverages(std::map<int, Mat> &histavgs) const {
    
    std::vector<Mat> hists;
    if(readHistograms(getHistogramAveragesFile(), hists) != true)
        return false;
    
    int histIdx = 0;
    for(std::map<int, int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); ++it) {
        histavgs[it->first] = hists.at(histIdx++);
    }

    return true;
}

// TODO: Replace with modelstorage
void xLBPH::mmapHistogramAverages() {
    
    std::cout << "loading histogram averages...\n";
    _histavgs.clear();
    String filename = getHistogramAveragesFile();
    int fd = open(filename.c_str(), O_RDONLY);
    if(fd < 0)
        CV_Error(Error::StsError, "Cannot open histogram file '"+filename+"'");

    unsigned char* mapPtr = (unsigned char*)mmap(NULL, getHistogramSize() * (int)_labelinfo.size() * SIZEOF_CV_32FC1, PROT_READ, MAP_PRIVATE, fd, 0);
    if(mapPtr == MAP_FAILED)
        CV_Error(Error::StsError, "Cannot mem map file '"+filename+"'");
    
    int idx = 0;
    for(std::map<int, int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); ++it) {
        Mat mat(1, getHistogramSize(), CV_32FC1, mapPtr + (getHistogramSize() * SIZEOF_CV_32FC1 * idx));
        _histavgs[it->first] = mat;
        idx++;
    }

    //std::cout << "test: " << matToHex(_histograms.at(2).at(0)) << "\n";
} 


//------------------------------------------------------------------------------
// Histogram Memory Mapping
//------------------------------------------------------------------------------
// TODO: Replace with modelstorage
void xLBPH::mmapHistograms() {

    //_histograms = std::map<int, std::vector<Mat>>();
    std::cout << "loading histograms...\n";
    _histograms.clear();
    for(std::map<int, int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); ++it) {
        // map histogram
        String filename = getHistogramFile(it->first);
        int fd = open(filename.c_str(), O_RDONLY);
        if(fd < 0)
            CV_Error(Error::StsError, "Cannot open histogram file '"+filename+"'");

        unsigned char* mapPtr = (unsigned char*)mmap(NULL, getHistogramSize() * it->second * SIZEOF_CV_32FC1, PROT_READ, MAP_PRIVATE, fd, 0);
        if(mapPtr == MAP_FAILED)
            CV_Error(Error::StsError, "Cannot mem map file '"+filename+"'");

        // make matricies
        for(int i = 0; i < it->second; i++) {
            Mat mat(1, getHistogramSize(), CV_32FC1, mapPtr + (getHistogramSize() * SIZEOF_CV_32FC1 * i));
            _histograms[it->first].push_back(mat);
        }

    }

}

void xLBPH::munmapHistograms() {
     
}

//------------------------------------------------------------------------------
// Clustering Functions
//------------------------------------------------------------------------------
void xLBPH::printMat(const Mat &mat, int label) const {

    int width = 11;
    printf("%4d_", label);
    for(int x = 0; x < mat.cols; x++) {
        printf("___%*d___|", width-4, x);
    }
    printf("\n");
    
    for(int y = 0; y < mat.rows; y++) {
        printf(" %2d | ", y);
        for(int x = 0; x < mat.cols; x++) {
            switch (mat.type()) {
                case CV_8SC1:  printf("%*d | ", width, mat.at<char>(x,y)); break; 
                case CV_8UC1:  printf("%*d | ", width, mat.at<unsigned char>(x,y)); break;
                case CV_16SC1: printf("%*d | ", width, mat.at<short>(x,y)); break; 
                case CV_16UC1: printf("%*d | ", width, mat.at<unsigned short>(x,y)); break;
                case CV_32SC1: printf("%*d | ", width, mat.at<int>(x,y)); break;
                case CV_32FC1: printf("%*.*f | ", width, width-2, mat.at<float>(x,y)); break;
                case CV_64FC1: printf("%*.*f | ", width, width-2, mat.at<double>(x,y)); break;
                default: printf(" type! | "); break;
            }
        }
        printf("\n");
    }
    printf("\n\n");
}

//------------------------------------------------------------------------------
// Histogram Clustering (Using Markov Clustering) 
//------------------------------------------------------------------------------

// Top level clustering function for training
void xLBPH::clusterHistograms(std::map<int, std::vector<cluster::cluster_t>> &clusters) const {
    /* What is Histogram Clustering?
     * The idea is to group like histograms together
     * Every label has a set of clusters
     * Every cluster has an average histogram and a set of histograms
     */
    
    std::vector<int> labels;
    for(std::map<int, int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); it++)
        labels.push_back(it->first);

    std::vector<std::pair<int, std::vector<cluster::cluster_t>>> allClusters;

    int count = 0;
    bool fail = false;
    tbb::parallel_for_each(_histograms.begin(), _histograms.end(), 
        [&](std::pair<int, std::vector<Mat>> it) {
    //for(std::map<int, std::vector<Mat>>::const_iterator it = _histograms.begin(); it != _histograms.end(); it++) {
        printf("\rClustering histograms %d / %d %20s", count, (int)_histograms.size(), " ");
        std::cout << std::flush;

        cluster::cluster_vars vars = {cluster_tierStep, 
                                    cluster_numTiers, 
                                    cluster_max_iterations,
                                    mcl_iterations, 
                                    mcl_expansion_power, 
                                    mcl_inflation_power, 
                                    mcl_prune_min};

        std::vector<cluster::cluster_t> labelClusters;
        cluster::clusterHistograms(_histograms.at(it.first), labelClusters, vars);
        

        if((int)labelClusters.size() <= 0) {
            printf("Found %3d clusters for label %5d from %5d histograms !!!\n", labelClusters.size(), it.first, (int)it.second.size());
            fail = true;
        }
        else {
            //printf("Found %3d clusters for label %5d from %5d histograms\n", labelClusters.size(), it.first, (int)it.second.size());
        } 

        //push all of the label clusters to the main clusters
        for(size_t i = 0; i < labelClusters.size(); i++) {
            clusters[it.first].push_back(labelClusters.at((int)i));
        }
        
        count++;
        printf("\rClustering histograms %d / %d %20s", count, (int)_histograms.size(), " ");
        std::cout << std::flush;
    //}
    });
    
    if(fail) {
        CV_Error(Error::StsError, "Error clustering histograms!!!"); 
    } 

    printf("Finished clustering histograms for %d %30s\n", (int)_histograms.size(), " ");

}



//------------------------------------------------------------------------------
// Standard Functions and File IO
//------------------------------------------------------------------------------
// TODO: Updated for modelstorage 
bool xLBPH::load() {
    
    if(!_model.isValidModel()) {
        return false; 
    } 

    AlgSettings alg;
    _model.loadMetadata(alg, _labelinfo);
    
    _radius = alg.radius;
    _neighbors = alg.neighbors;
    _grid_x = alg.grid_x;
    _grid_y = alg.grid_y;

    _model.mmapLabelHistograms(_labelinfo, _histograms);
    _model.mmapLabelAverages(_labelinfo, _histavgs);
    _model.mmapClusters(_labelinfo, _clusters);

    return true;

    /*
    // load data from the info file
    std::cout << "loading info file...\n";
    FileStorage infofile(getInfoFile(), FileStorage::READ);
    if (!infofile.isOpened())
        CV_Error(Error::StsError, "File '"+getInfoFile()+"' can't be opened for reading!");
    
    // alg settings
    infofile["radius"] >> _radius;
    infofile["neighbors"] >> _neighbors;
    infofile["grid_x"] >> _grid_x;
    infofile["grid_y"] >> _grid_y;
    
    // label_info
    std::vector<int> labels;
    std::vector<int> numhists;
    FileNode label_info = infofile["label_info"];
    label_info["labels"] >> labels;
    label_info["numhists"] >> numhists;
  
    CV_Assert(labels.size() == numhists.size());
    _labelinfo = std::map<int, int>(); 
    for(size_t idx = 0; idx < labels.size(); idx++) {
        _labelinfo[labels.at((int)idx)] = numhists.at((int)idx);
    }
    infofile.release();

    // mem map histograms
    mmapHistograms();
    mmapHistogramAverages();
    */
}


// TODO: Updated for modelstorage 
void xLBPH::load(const String &modelpath) {
    
    // set our model path to the filename
    setModelPath(modelpath);

    // load the model
    load();    
}


// See FaceRecognizer::load.
/* TODO: Rewrite for xLBPH
 * sets modelpath
 * loads alg settings from infofile
 * loads labelinfo from infofile
 */
void xLBPH::load(const FileStorage& fs) {
    if (!fs.isOpened())
        CV_Error(Error::StsError, "File can't be opened for writing!");
}

// See FaceRecognizer::save.
/* TODO: Rewrite for xLBPH
 * wha does this do?
 * write infofile
 */
void xLBPH::save(FileStorage& fs) const {
    if (!fs.isOpened())
        CV_Error(Error::StsError, "File can't be opened for reading!");
}

void xLBPH::train(InputArrayOfArrays _in_src, InputArray _in_labels) {
    this->train(_in_src, _in_labels, false);
}

void xLBPH::update(InputArrayOfArrays _in_src, InputArray _in_labels) {
    // got no data, just return
    if(_in_src.total() == 0)
        return;

    this->train(_in_src, _in_labels, true);
}


//------------------------------------------------------------------------------
// xLBPH
//------------------------------------------------------------------------------

// TODO: Can we move this math shit into it's own file for organization sake?

template <typename _Tp> static
void olbp_(InputArray _src, OutputArray _dst) {
    // get matrices
    Mat src = _src.getMat();
    // allocate memory for result
    _dst.create(src.rows-2, src.cols-2, CV_8UC1);
    Mat dst = _dst.getMat();
    // zero the result matrix
    dst.setTo(0);
    // calculate patterns
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            _Tp center = src.at<_Tp>(i,j);
            unsigned char code = 0;
            code |= (src.at<_Tp>(i-1,j-1) >= center) << 7;
            code |= (src.at<_Tp>(i-1,j) >= center) << 6;
            code |= (src.at<_Tp>(i-1,j+1) >= center) << 5;
            code |= (src.at<_Tp>(i,j+1) >= center) << 4;
            code |= (src.at<_Tp>(i+1,j+1) >= center) << 3;
            code |= (src.at<_Tp>(i+1,j) >= center) << 2;
            code |= (src.at<_Tp>(i+1,j-1) >= center) << 1;
            code |= (src.at<_Tp>(i,j-1) >= center) << 0;
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
}

//------------------------------------------------------------------------------
// cv::elbp
//------------------------------------------------------------------------------
template <typename _Tp> static
inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors) {
    //get matrices
    Mat src = _src.getMat();
    // allocate memory for result
    _dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    Mat dst = _dst.getMat();
    // zero
    dst.setTo(0);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius * cos(2.0*CV_PI*n/static_cast<float>(neighbors)));
        float y = static_cast<float>(-radius * sin(2.0*CV_PI*n/static_cast<float>(neighbors)));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                // calculate interpolated value
                float t = static_cast<float>(w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx));
                // floating point precision, so check some machine-dependent epsilon
                dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || (std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}

static void elbp(InputArray src, OutputArray dst, int radius, int neighbors)
{
    int type = src.type();
    switch (type) {
    case CV_8SC1:   elbp_<char>(src,dst, radius, neighbors); break;
    case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors); break;
    case CV_16SC1:  elbp_<short>(src,dst, radius, neighbors); break;
    case CV_16UC1:  elbp_<unsigned short>(src,dst, radius, neighbors); break;
    case CV_32SC1:  elbp_<int>(src,dst, radius, neighbors); break;
    case CV_32FC1:  elbp_<float>(src,dst, radius, neighbors); break;
    case CV_64FC1:  elbp_<double>(src,dst, radius, neighbors); break;
    default:
        String error_msg = format("Using Original Local Binary Patterns for feature extraction only works on single-channel images (given %d). Please pass the image data as a grayscale image!", type);
        CV_Error(Error::StsNotImplemented, error_msg);
        break;
    }
}

static Mat
histc_(const Mat& src, int minVal=0, int maxVal=255, bool normed=false)
{
    Mat result;
    // Establish the number of bins.
    int histSize = maxVal-minVal+1;
    // Set the ranges.
    float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal+1) };
    const float* histRange = { range };
    // calc histogram
    calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &histRange, true, false);
    // normalize
    if(normed) {
        result /= (int)src.total();
    }
    return result.reshape(1,1);
}

static Mat histc(InputArray _src, int minVal, int maxVal, bool normed)
{
    Mat src = _src.getMat();
    switch (src.type()) {
        case CV_8SC1:
            return histc_(Mat_<float>(src), minVal, maxVal, normed);
            break;
        case CV_8UC1:
            return histc_(src, minVal, maxVal, normed);
            break;
        case CV_16SC1:
            return histc_(Mat_<float>(src), minVal, maxVal, normed);
            break;
        case CV_16UC1:
            return histc_(src, minVal, maxVal, normed);
            break;
        case CV_32SC1:
            return histc_(Mat_<float>(src), minVal, maxVal, normed);
            break;
        case CV_32FC1:
            return histc_(src, minVal, maxVal, normed);
            break;
        default:
            CV_Error(Error::StsUnmatchedFormats, "This type is not implemented yet."); break;
    }
    return Mat();
}


static Mat spatial_histogram(InputArray _src, int numPatterns,
                             int grid_x, int grid_y, bool /*normed*/)
{
    Mat src = _src.getMat();
    // calculate LBP patch size
    int width = src.cols/grid_x;
    int height = src.rows/grid_y;
    // allocate memory for the spatial histogram
    Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
    // return matrix with zeros if no data was given
    if(src.empty())
        return result.reshape(1,1);
    // initial result_row
    int resultRowIdx = 0;
    // iterate through grid
    for(int i = 0; i < grid_y; i++) {
        for(int j = 0; j < grid_x; j++) {
            Mat src_cell = Mat(src, Range(i*height,(i+1)*height), Range(j*width,(j+1)*width));
            Mat cell_hist = histc(src_cell, 0, (numPatterns-1), true);

            // copy to the result matrix
            Mat result_row = result.row(resultRowIdx);
            cell_hist.reshape(1,1).convertTo(result_row, CV_32FC1);

            // free memory
            src_cell.release();
            cell_hist.release();
   
            // increase row count in result matrix
            resultRowIdx++;
        }
    }
    // return result as reshaped feature vector
    return result.reshape(1,1);
}

//------------------------------------------------------------------------------
// wrapper to cv::elbp (extended local binary patterns)
//------------------------------------------------------------------------------

static Mat elbp(InputArray src, int radius, int neighbors) {
    Mat dst;
    elbp(src, dst, radius, neighbors);
    return dst;
}



//------------------------------------------------------------------------------
// Multithreading 
//------------------------------------------------------------------------------
// REMOVE: Replaced by TBB
template <typename _Tp> static
void splitVector(const std::vector<_Tp> &src, std::vector<std::vector<_Tp>> &dst, int numParts) {
    int step = (int)src.size() / numParts;
    typename std::vector<_Tp>::const_iterator start = src.begin();
    for(int i = 0; i < numParts; i++) {
        typename std::vector<_Tp>::const_iterator end;
        if(i < numParts - 1) {
            end = start + step;
            if(end > src.end())
                end = src.end();
        }
        else {
            end = src.end(); 
        }
        dst.push_back(std::vector<_Tp>(start, end));
        start += step;
    }
}

/* performs a multithreaded calculation on the given <src> vector, putting the
 * results into the given <dst> vector. Attempts to use <numThreads> to split
 * up the <src> vector. Calls <calcFunc> to do the calculation.
 * TODO:
 * If numThreads is:
 * - (== 1 or == -1) then will simply call <calcFunc> without making a new 
 *      thread or splitting up the <src> vector 
 *      -> does not dispatch new threads
 * - (> 1) then will attempt to use <numThreads> up to _numThreads threads to 
 *      split up the <src> vector 
 *      -> will dispatch new threads unless _numThreads is set to <= 1
 * - (== 0) then will attempt to figure out an "optimal" number of threads to 
 *      use based on the size of src up to _numThreads 
 *       -> might dispatch threads, depends on input size and _numThreads, will
 *          either behave as (== 1 or == -1) or as (> 1)
 * - (< -1) then will attempt to figure out an "optimal" number of threads to
 *      use based on the size of src up to either the absolute value of 
 *      <numThreads> or up to _numThreads. For example if <numThreads> is -4 then
 *      will only use up to 4 threads.
 *       -> will dispatch new threads unless _numThreads is set to <= 1
 * 
 * If numThreads is greater than the <src> size it will be capped to the size of
 * <src>, will not try to dispatch more threads than tasks available
 */
// REMOVE: Replaced by TBB
template <typename S, typename D>
void xLBPH::performMultithreadedCalc(const std::vector<S> &src, std::vector<D> &dst, int numThreads, void (xLBPH::*calcFunc)(const std::vector<S> &src, std::vector<D> &dst) const) const {
    if(numThreads > (int)src.size())
        numThreads = (int)src.size();

    if(numThreads <= 0)
        CV_Error(Error::StsBadArg, "numThreads must greater than 0!");
    else if(numThreads == 1)
        (this->*calcFunc)(src, dst);
    else
    {
        //split src
        std::vector<std::vector<S>> splitSrc;
        splitVector<S>(src, splitSrc, numThreads);

        //dispatch threads
        std::vector<std::vector<D>> splitDst(numThreads, std::vector<D>(0));
        std::vector<std::thread> threads;
        for(int i = 0; i < numThreads; i++) {
            threads.push_back(std::thread(calcFunc, this, std::ref(splitSrc.at(i)), std::ref(splitDst.at(i))));
        }
        
        //wait for threads
        for(size_t idx = 0; idx < threads.size(); idx++) {
            threads.at((int)idx).join();
        }
        
        //recombine splitDst 
        for(size_t idx = 0; idx < splitDst.size(); idx++) {
            std::vector<D> threadDst = splitDst.at((int)idx);
            for(size_t threadidx = 0; threadidx < threadDst.size(); threadidx++) {
                dst.push_back(threadDst.at((int)threadidx));
            } 
        }

    }
}

// REMOVE: Repalced by TBB
template <typename Q, typename S, typename D>
void xLBPH::performMultithreadedComp(const Q &query, const std::vector<S> &src, std::vector<D> &dst, int numThreads, void (xLBPH::*compFunc)(const Q &query, const std::vector<S> &src, std::vector<D> &dst) const) const {
    if(numThreads <= 0)
        CV_Error(Error::StsBadArg, "numThreads must greater than 0!");
    else if(numThreads == 1)
        (this->*compFunc)(query, src, dst);
    else
    {
        //split src
        std::vector<std::vector<S>> splitSrc;
        splitVector<S>(src, splitSrc, numThreads);

        //dispatch threads
        std::vector<std::vector<D>> splitDst(numThreads, std::vector<D>(0));
        std::vector<std::thread> threads;
        for(int i = 0; i < numThreads; i++) {
            threads.push_back(std::thread(compFunc, this, std::ref(query), std::ref(splitSrc.at(i)), std::ref(splitDst.at(i))));
        }
        
        //wait for threads
        for(size_t idx = 0; idx < threads.size(); idx++) {
            threads.at((int)idx).join();
        }
    
        //recombine splitDst 
        for(size_t idx = 0; idx < splitDst.size(); idx++) {
            std::vector<D> threadDst = splitDst.at((int)idx);
            for(size_t threadidx = 0; threadidx < threadDst.size(); threadidx++) {
                dst.push_back(threadDst.at((int)threadidx));
            } 
        }

    }
}



//------------------------------------------------------------------------------
// Training Functions
//------------------------------------------------------------------------------



// Main training function
void xLBPH::train(InputArrayOfArrays _in_src, InputArray _in_labels, bool preserveData) {

    if(_in_src.kind() != _InputArray::STD_VECTOR_MAT && _in_src.kind() != _InputArray::STD_VECTOR_VECTOR) {
        String error_message = "The images are expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< std::vector<...>>).";
        CV_Error(Error::StsBadArg, error_message);
    }
    if(_in_src.total() == 0) {
        String error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(Error::StsUnsupportedFormat, error_message);
    } else if(_in_labels.getMat().type() != CV_32SC1) {
        String error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _in_labels.type());
        CV_Error(Error::StsUnsupportedFormat, error_message);
    }

    // get the vector of matrices
    std::vector<Mat> src;
    _in_src.getMatVector(src);
    // get the label matrix
    Mat labels = _in_labels.getMat();
    // check if data is well- aligned
    if(labels.total() != src.size()) {
        String error_message = format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", src.size(), labels.total());
        CV_Error(Error::StsBadArg, error_message);
    }
    
    // Get all of the labels
    std::vector<int> allLabels;
    for(size_t labelIdx = 0; labelIdx < labels.total(); labelIdx++) {
       allLabels.push_back(labels.at<int>((int)labelIdx));
    }
     
    std::cout << "Organizing images by label...\n";
    // organize the mats and labels
    std::map<int, std::vector<Mat>> labelImages;
    for(size_t matIdx = 0; matIdx < src.size(); matIdx++) {
        labelImages[allLabels.at((int)matIdx)].push_back(src.at((int)matIdx));
    }
    std::cout << "Organized images for " << labelImages.size() << " labels.\n";
    
    
    /*
    if(!preserveData)
    {
        // create model directory
        // check if the model directory already exists
        if(exists(getModelPath())) {
            // model directory exists
            // check if the directory is actually a model
            if(exists(getInfoFile()) && exists(getHistogramsDir())) {
                // is a model dir so overwrite  
                system(("rm -r " + getModelPath()).c_str());     
            }
            else {
                // is not a model dir so error
                CV_Error(Error::StsError, "Given model path at '" + getModelPath() +"' already exists and doesn't look like an xLBPH model directory; refusing to overwrite for data safety.");
            }
        }
        
        // create the model directories
        system(("mkdir " + getModelPath()).c_str()); 
        system(("mkdir " + getHistogramsDir()).c_str());
    }
    */


    std::vector<int> uniqueLabels;
    std::vector<int> numhists;

    // start training
    if(preserveData)
    {
        // not overwriting, updating
        
        // make sure our current model is valid, don't want to update an invalid model
        if(!_model.isValidModel()) {
            CV_Error(Error::StsError, "Model is malformed, won't update a bad model!");
        } 

        int count = 0;
        tbb::parallel_for_each(labelImages.begin(), labelImages.end(), 
            [&](std::pair<int, std::vector<Mat>> it) {
                std::cout << "Calculating histograms for label " << count++ << " / " << labelImages.size() << " [" << it.first << "]\r" << std::flush;
            
                std::vector<Mat> imgs = it.second;
              
                tbb::concurrent_vector<Mat> concurrent_hists;
                
                tbb::parallel_for_each(imgs.begin(), imgs.end(),
                    [&](Mat img) {
                           
                        Mat lbp_image = elbp(img, _radius, _neighbors);

                        // get spatial histogram from this lbp image
                        Mat p = spatial_histogram(
                                lbp_image, // lbp_image
                                static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), // number of possible patterns
                                _grid_x, // grid size x
                                _grid_y, // grid size y
                                true);
                        
                        concurrent_hists.push_back(p);
                    } 
                );

                uniqueLabels.push_back(it.first);
                numhists.push_back((int)imgs.size());
                std::vector<Mat> hists(concurrent_hists.begin(), concurrent_hists.end());
                if(!_model.updateLabelHistograms(it.first, hists)) {
                    CV_Error(Error::StsError, "Failed to update label histograms!");
                } 
                //writeHistograms(getHistogramFile(it.first), hists, true);
                hists.clear();
            }
        );
    }
    else {
        // overwriting
        
        // attempt to create the new model
        if(!_model.create(true)) {
            CV_Error(Error::StsError, "Failed to create new model!");
        } 

        std::cout << "Multithreaded label calcs\n";
        std::vector<std::pair<int, std::vector<Mat>>> labelImagesVec(labelImages.begin(), labelImages.end());
        tbb::concurrent_vector<std::pair<int, int>> concurrent_labelInfoVec;
        
        int count = 0;
        tbb::parallel_for_each(labelImagesVec.begin(), labelImagesVec.end(), 
            [&](std::pair<int, std::vector<Mat>> it) {
                std::cout << "Calculating histograms " << count++ << " / " << (int)labelImagesVec.size() << "                       \r" << std::flush;
                
                int label = it.first;
                std::vector<Mat> imgs = it.second;
                //std::pair<int, int> info(label, (int)imgs.size());
                //concurrent_labelInfoVec.push_back(info);
                concurrent_labelInfoVec.push_back(std::pair<int, int>(label, (int)imgs.size()));

                tbb::concurrent_vector<Mat> concurrent_hists;
                
                tbb::parallel_for_each(imgs.begin(), imgs.end(),
                    [&](Mat img) {
                           
                        Mat lbp_image = elbp(img, _radius, _neighbors);

                        // get spatial histogram from this lbp image
                        Mat p = spatial_histogram(
                                lbp_image, // lbp_image
                                static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), // number of possible patterns
                                _grid_x, // grid size x
                                _grid_y, // grid size y
                                true);
                        
                        concurrent_hists.push_back(p);
                    } 
                );
                
                std::vector<Mat> hists(concurrent_hists.begin(), concurrent_hists.end());
                if(!_model.saveLabelHistograms(label, hists)) {
                    CV_Error(Error::StsError, "Failed to save label histograms!");
                } 
                //writeHistograms(getHistogramFile(label), hists, false);
            }
        );

        std::vector<std::pair<int, int>> labelInfoVec(concurrent_labelInfoVec.begin(), concurrent_labelInfoVec.end());
        for(size_t idx = 0; idx < labelInfoVec.size(); idx++) {
            uniqueLabels.push_back(labelInfoVec.at((int)idx).first);
            numhists.push_back(labelInfoVec.at((int)idx).second);
        }
    }
    
    std::cout << "Finished calculating histograms for " << labelImages.size() << " labels.                                      \n";

    // set _labelinfo
    _labelinfo = std::map<int, int>(); // if _labelinfo was set then clear it
    for(size_t labelIdx = 0; labelIdx < uniqueLabels.size(); labelIdx++) {
        _labelinfo[uniqueLabels.at((int)labelIdx)] = numhists.at((int)labelIdx);
    }
    
    // write metadata
    //std::cout << "Writing model metadata...\n";
    AlgSettings alg = {_radius, _neighbors, _grid_x, _grid_y};
    if(!_model.writeMetadata(alg, _labelinfo)) {
        CV_Error(Error::StsError, "Failed to write model metadata!");
    } 

    /*
    String infofilepath(_modelpath + "/" + getModelName() + ".yml");
    FileStorage infofile(infofilepath, FileStorage::WRITE);
    if (!infofile.isOpened())
        CV_Error(Error::StsError, "File can't be opened for writing!");

    infofile << "radius" << _radius;
    infofile << "neighbors" << _neighbors;
    infofile << "grid_x" << _grid_x;
    infofile << "grid_y" << _grid_y;
    infofile << "numlabels" << (int)labelImages.size();
    //infofile << "labels" << unique_labels;
    infofile << "label_info" << "{";
    infofile << "labels" << uniqueLabels;
    infofile << "numhists" << numhists;
    infofile << "}";
    infofile.release();
    */

    // Load histograms 
    _model.mmapLabelHistograms(_labelinfo, _histograms);

    std::cout << "Calculating label averages...\n";
    std::vector<Mat> histavgs;
    calcHistogramAverages(histavgs);

    //std::cout << "Writing label averages...\n";
    if(!_model.saveLabelAverages(histavgs)) {
        CV_Error(Error::StsError, "Failed to write label averages!");
    } 
   
    // Load label averages
    _model.mmapLabelAverages(_labelinfo, _histavgs);

    //mmapHistograms();
    //mmapHistogramAverages();    


    if(_useClusters) {
        std::map<int, std::vector<cluster::cluster_t>> clusters;
        clusterHistograms(clusters);
        
        if(!_model.saveClusters(clusters)) {
            CV_Error(Error::StsError, "Failed to write clusters!");
        } 

        _model.mmapClusters(_labelinfo, _clusters);
    }

    //load();

    std::cout << "Training complete\n";
}

//------------------------------------------------------------------------------
// Prediction Functions 
//------------------------------------------------------------------------------



//void xLBPH::predict_avg_clustering(InputArray _query, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::set<int> &labels) const {
void xLBPH::predict_avg_clustering(InputArray _query, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::vector<int> &labels) const {
//void xLBPH::predict_avg_clustering(InputArray _query, int &minClass, double &minDist) const {

    if(!_useClusters) {
        CV_Error(Error::StsError, "Cannot make prediction using clusters, clustering disabled!"); 
        return;
    }

    Mat query = _query.getMat();
    
    
    tbb::concurrent_vector<std::pair<double, int>> bestlabels;

    tbb::parallel_for_each(labels.begin(), labels.end(),
        [&bestlabels, &query, this](int label) {
            if(_histavgs.find(label) != _histavgs.end()) {
                bestlabels.push_back(std::pair<double, int>(compareHist(_histavgs.at(label), query, COMP_ALG), label));
            } 
        }
    );
    std::sort(bestlabels.begin(), bestlabels.end());

    // figure out how many labels to check
    int numLabelsToCheck = (int)((int)labels.size() * labelsToCheckRatio);
    if(numLabelsToCheck < minLabelsToCheck)
        numLabelsToCheck = minLabelsToCheck;
    if(numLabelsToCheck > (int)bestlabels.size())
        numLabelsToCheck = (int)bestlabels.size();


    //tbb::concurrent_vector<std::pair<int, std::vector<Mat>>> labelhists;
    tbb::concurrent_vector<std::pair<int, cluster::idx_cluster_t>> labelhists;
    tbb::parallel_for(0, numLabelsToCheck, 1, 
        [&](int i) {
            
            int label = bestlabels.at(i).second;
            std::vector<std::pair<Mat, cluster::idx_cluster_t>> labelClusters = _clusters.at(label);
            tbb::concurrent_vector<std::pair<double, int>> clusterDists;
            
            // find the best clusters
            tbb::parallel_for(0, (int)labelClusters.size(), 1,
                [&i, &labelClusters, &clusterDists, &query](int clusterIdx) {
                    clusterDists.push_back(std::pair<double, int>(compareHist(labelClusters.at(clusterIdx).first, query, COMP_ALG), clusterIdx));
                } 
            );
            std::sort(clusterDists.begin(), clusterDists.end());

            // figure out how many labels to check
            int numClustersToCheck = (int)((int)clusterDists.size() * clustersToCheckRatio);
            if(numClustersToCheck < minClustersToCheck)
                numClustersToCheck = minClustersToCheck;
            if(numClustersToCheck > (int)clusterDists.size())
                numClustersToCheck = (int)clusterDists.size();
            
            // group together all thehistograms in this labels best clusters 
            cluster::idx_cluster_t combinedClusters;
            for(size_t bestIdx = 0; bestIdx < clusterDists.size() && (int)bestIdx < numClustersToCheck; bestIdx++) {
                
                int labelClustersIdx = clusterDists.at((int)bestIdx).second;
                cluster::idx_cluster_t cluster = labelClusters.at(labelClustersIdx).second;
                //std::set<Mat> cluster = labelClusters.at(labelClustersIdx).second; 

                for(size_t clusterIdx = 0; clusterIdx < cluster.size(); clusterIdx++) {
                   combinedClusters.push_back(cluster.at((int)clusterIdx));
                }
            }

            labelhists.push_back(std::pair<int, cluster::idx_cluster_t>(label, combinedClusters));
        
        }
    );

    //printf(" - Calculating distances for best clusters...\n");
   
    // check best labels by cluster
    tbb::concurrent_vector<std::pair<int, std::vector<double>>> labeldists;
    tbb::parallel_for_each(labelhists.begin(), labelhists.end(),
        [&labelhists, &labeldists, &query, this](std::pair<int, cluster::idx_cluster_t> it) {
            
            cluster::idx_cluster_t cluster = it.second;
            std::vector<Mat> allHists = _histograms.at(it.first);
            std::vector<Mat> hists;
            for(size_t i = 0; i < cluster.size(); i++) {
                hists.push_back(allHists.at(cluster.at(i)));
            }
            
            //printf("For label %d looking at %d histograms out of %d total\n", it.first, (int)hists.size(), (int)allHists.size());

            //std::vector<Mat> hists = it.second;
            tbb::concurrent_vector<double> dists;

            tbb::parallel_for_each(hists.begin(), hists.end(), 
                [&labeldists, &dists, &query](Mat hist) {
                    dists.push_back(compareHist(hist, query, COMP_ALG));
                } 
            );
            std::sort(dists.begin(), dists.end());
            labeldists.push_back(std::pair<int, std::vector<double>>(it.first, std::vector<double>(dists.begin(), dists.end())));
        } 
    ); 

    //tbb::concurrent_vector<std::pair<double, int>> bestpreds;
    for(size_t idx = 0; idx < labeldists.size(); idx++) {
        std::vector<double> dists = labeldists.at((int)idx).second;
        bestpreds.push_back(std::pair<double, int>(dists.at(0), labeldists.at((int)idx).first));
    }
    std::sort(bestpreds.begin(), bestpreds.end());
    
} 

//void xLBPH::predict_avg(InputArray _query, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::set<int> &labels) const {
void xLBPH::predict_avg(InputArray _query, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::vector<int> &labels) const {
    Mat query = _query.getMat();

    
    tbb::concurrent_vector<std::pair<double, int>> bestlabels;
    
    //printf("Finding bestlabels...\n");
    tbb::parallel_for_each(labels.begin(), labels.end(),
        [&bestlabels, &query, this](int label) {
            if(_histavgs.find(label) != _histavgs.end()) {
                bestlabels.push_back(std::pair<double, int>(compareHist(_histavgs.at(label), query, COMP_ALG), label));
            } 
        }
    );

    std::sort(bestlabels.begin(), bestlabels.end());


    // figure out how many labels to check
    int numLabelsToCheck = (int)((int)labels.size() * labelsToCheckRatio);
    if(numLabelsToCheck < minLabelsToCheck)
        numLabelsToCheck = minLabelsToCheck;
    if(numLabelsToCheck > (int)bestlabels.size())
        numLabelsToCheck = (int)bestlabels.size();

    //printf("Checking %d best labels...\n", numLabelsToCheck);

    //tbb::concurrent_vector<std::pair<double, int>> bestpreds;
    tbb::parallel_for(0, numLabelsToCheck, 1, 
        [&](int i) {
            tbb::concurrent_vector<double> dists;
            std::vector<Mat> hists = _histograms.at(bestlabels.at(i).second);

            tbb::parallel_for_each(hists.begin(), hists.end(),
                [&dists, &query](Mat hist) {
                    dists.push_back(compareHist(hist, query, COMP_ALG));
                }
            );
            std::sort(dists.begin(), dists.end());
            bestpreds.push_back(std::pair<double, int>(dists.at(0), bestlabels.at(i).second));
        } 
    );
    std::sort(bestpreds.begin(), bestpreds.end());
} 


//void xLBPH::predict_std(InputArray _query, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::set<int> &labels) const {
void xLBPH::predict_std(InputArray _query, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::vector<int> &labels) const {
    Mat query = _query.getMat();

    tbb::parallel_for_each(labels.begin(), labels.end(),
        [&bestpreds, &query, this](int label) {
            if(_histograms.find(label) != _histograms.end()) {
                std::vector<Mat> hists = _histograms.at(label);
                
                tbb::concurrent_vector<double> dists;

                tbb::parallel_for_each(hists.begin(), hists.end(), 
                    [&dists, &query](Mat hist) {
                        dists.push_back(compareHist(hist, query, COMP_ALG));
                    } 
                );

                std::sort(dists.begin(), dists.end());
                
                bestpreds.push_back(std::pair<double, int>(dists.at(0), label));
            } 
        }
    );
    std::sort(bestpreds.begin(), bestpreds.end());
}


void xLBPH::predictMulti(InputArray _src, OutputArray _preds, int numPreds, InputArray _labels) const {
    CV_Assert((int)_labelinfo.size() > 0);
    CV_Assert((int)_histograms.size() > 0);
    

    Mat src = _src.getMat();
    // get the spatial histogram from input image
    Mat lbp_image = elbp(src, _radius, _neighbors);
    Mat query = spatial_histogram(
            lbp_image, /* lbp_image */
            static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), /* number of possible patterns */
            _grid_x, /* grid size x */
            _grid_y, /* grid size y */
            true /* normed histograms */);
    
    //printf("Extracting labels to consider...\n");
    // Gets the list of labels to check
    Mat labelsMat = _labels.getMat();
    std::vector<int> labels;
    for(size_t labelIdx = 0; labelIdx < labelsMat.total(); labelIdx++)
        labels.push_back(labelsMat.at<int>((int)labelIdx));
    //printf("Considering %d labels out of %d labels total...\n", (int)labels.size(), (int)_labelinfo.size());


    //printf("Calling prediction algorithm...\n");
    tbb::concurrent_vector<std::pair<double, int>> bestpreds;
    switch(_algToUse) {
        case 1: predict_avg(query, bestpreds, labels); break;
        case 2: predict_avg_clustering(query, bestpreds, labels); break;
        default: predict_std(query, bestpreds, labels); break;
    }
    
    
    //printf("Compiling prediction results...\n");
    if(bestpreds.size() < numPreds)
        numPreds = (int)bestpreds.size();

    _preds.create(numPreds, 2, CV_64FC1);
    Mat preds = _preds.getMat();

    //std::cout << "\nBest Prediction by PID:\n";
    int i = 0;
    for(tbb::concurrent_vector<std::pair<double, int>>::const_iterator it = bestpreds.begin(); it != bestpreds.end(); ++it) {
        //printf("[%d, %f]\n", it->second, it->first);

        if(i < numPreds) {
            preds.at<double>(i, 0) = it->second;
            preds.at<double>(i, 1) = it->first;
            i++;
        }
    }
    
}

void xLBPH::predictAll(std::vector<Mat> &_src, std::vector<Mat> &_preds, int numPreds, InputArray _labels) const {

    // setup for concurrency
    tbb::concurrent_vector<Mat> images;
    for(std::vector<Mat>::const_iterator it = _src.begin(); it != _src.end(); it++) {
        images.push_back(*it);
    }
    
    // begin prediction
    tbb::concurrent_vector<Mat> allPreds;
    tbb::parallel_for_each(images.begin(), images.end(),
        [&allPreds, &_labels, &numPreds, this](Mat image) {

            Mat pred;
            predictMulti(image, pred, numPreds, _labels);
            allPreds.push_back(pred);

        }
    );

    // set output array
    for(tbb::concurrent_vector<Mat>::const_iterator it = allPreds.begin(); it != allPreds.end(); ++it) {
        _preds.push_back(*it);
    }

}


void xLBPH::predictMulti(InputArray _src, OutputArray _preds, int numPreds) const {
    std::vector<int> labels;
    for(std::map<int,int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); it++)
        labels.push_back(it->first);

    predictMulti(_src, _preds, numPreds, labels);
}

void xLBPH::predict(InputArray _src, int &minClass, double &minDist) const {
    Mat dst;
    predictMulti(_src, dst, 1);
    minClass = (int)dst.at<float>(0,0);
    minDist = dst.at<float>(1,0);
}

int xLBPH::predict(InputArray _src) const {
    int label;
    double dummy;
    predict(_src, label, dummy);
    return label;
}

Ptr<xLBPHFaceRecognizer> createxLBPHFaceRecognizer(int radius, int neighbors,
                                                   int grid_x, int grid_y, 
                                                   double threshold, 
                                                   String modelpath)
{
    return makePtr<xLBPH>(radius, neighbors, grid_x, grid_y, threshold, modelpath);
}

}}
