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
    std::map<int, std::vector<clstr::cluster_t>> _clusters;

    // defines what prediction algorithm to use
    int _algToUse;
    
    bool _useClusters;

    //--------------------------------------------------------------------------
    // Multithreading
    //--------------------------------------------------------------------------
    template <typename S, typename D>
    void performMultithreadedCalc(const std::vector<S> &src, std::vector<D> &dst, int numThreads, void (xLBPH::*calcFunc)(const std::vector<S> &src, std::vector<D> &dst) const) const;
    template <typename Q, typename S, typename D>
    void performMultithreadedComp(const Q &query, const std::vector<S> &src, std::vector<D> &dst, int numThreads, void (xLBPH::*compFunc)(const Q &query, const std::vector<S> &src, std::vector<D> &dst) const) const;
    
    int _numThreads;

    int getMaxThreads() const;
    int getLabelThreads() const; // threads that iterate through labels
    int getHistThreads() const; // threads that iterate through histograms

    //--------------------------------------------------------------------------
    // Model Training Function
    //--------------------------------------------------------------------------
    // Computes a LBPH model with images in src and
    // corresponding labels in labels, possibly preserving
    // old model data.
    void train(InputArrayOfArrays src, InputArray labels, bool preserveData);

    void calculateLabels(const std::vector<std::pair<int, std::vector<Mat>>> &labelImages, std::vector<std::pair<int, int>> &labelinfo) const;
    void calculateHistograms(const std::vector<Mat> &src, std::vector<Mat> &dst) const;
    //void calculateHistograms_multithreaded(const std::vector<Mat> &images, std::vector<Mat> &histsdst, bool makeThreads = false);
    //void calculateHistograms_multithreaded(const std::vector<Mat> &images, std::vector<Mat> &histsdst);
    //void trainLabel_multithreaded(std::vector<Mat> &images, std::vector<Mat> &histsdst);

    //--------------------------------------------------------------------------
    // Prediction Functions
    //--------------------------------------------------------------------------
    /*
    void predict_std(InputArray _src, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::set<int> &labels) const;
    void predict_avg(InputArray _src, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::set<int> &labels) const;
    void predict_avg_clustering(InputArray _src, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::set<int> &labels) const;
    */ 

    void predict_std(InputArray _src, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::vector<int> &labels) const;
    void predict_avg(InputArray _src, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::vector<int> &labels) const;
    void predict_avg_clustering(InputArray _src, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::vector<int> &labels) const;
    /*
    void predict_avg(InputArray _src, int &label, double &dist) const;
    void predict_avg_clustering(InputArray _query, int &minClass, double &minDist) const;
    */

    void compareLabelHistograms(const Mat &query, const std::vector<std::pair<int, std::vector<Mat>>> &labelhists, std::vector<std::pair<int, std::vector<double>>> &labeldists) const;
    //void compareLabelWithQuery(const Mat &query, const std::vector<int> &labels, std::vector<std::vector<double>> &labeldists) const;
    void compareHistograms(const Mat &query, const std::vector<Mat> &hists, std::vector<double> &dists) const;
    
    int minLabelsToCheck = 10;
    double labelsToCheckRatio = 0.05;

    int minClustersToCheck = 5;
    double clustersToCheckRatio = 0.5;

    //void predict_cluster(InputArray _src, int &label, double &dist) const;

    //--------------------------------------------------------------------------
    // Managing Histogram Binary Files 
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
    bool calcHistogramAverages() const;
    void calcHistogramAverages_thread(const std::vector<int> &labels, std::vector<Mat> &avgsdst) const;
    bool loadHistogramAverages(std::map<int, Mat> &histavgs) const;
    void mmapHistogramAverages();

    //--------------------------------------------------------------------------
    // Histogram Clustering and Markov Clustering
    //--------------------------------------------------------------------------
    void clusterHistograms();
    void cluster_calc_weights(Mat &dists, Mat &weights, double tierStep, int numTiers);
    void cluster_dists(Mat &dists, Mat &mclmat, double r);
    void cluster_interpret(Mat &mclmat, std::vector<std::set<int>> &clusters);
    double cluster_ratio(std::vector<std::set<int>> &clusters);
    void cluster_find_optimal(Mat &dists, std::vector<std::set<int>> &clusters);
    
    void cluster_label(int label, std::vector<std::pair<Mat, std::vector<Mat>>> &matClusters);
    //void cluster_label(int label, std::vector<std::set<int>> &clusters);

    //void cluster_label(int label, std::vector<std::pair<Mat, std::vector<Mat>>> &clusters);

    void printMat(const Mat &mat, int label) const;
    
    // Histogram Clustering - Settings
    double cluster_tierStep = 0.01; // Sets how large a tier is, default is 0.01 or 1%
    int cluster_numTiers = 10; // Sets how many tiers to keep, default is 10, or 10% max tier
    int cluster_max_iterations = 5;

    // Markov Clustering Algorithm (MCL)- Settings
    /* Sets the number of MCL iterations, default is 10
     * If 0 then iterates until no change is found
     */
    int mcl_iterations = 10;
    int mcl_expansion_power = 2; // Sets the expansion power exponent, default is 2
    double mcl_inflation_power = 2; // Sets the inflation power exponent, default is 2 
    double mcl_prune_min = 0.001; // Sets the minimum value to prune, any values below this are set to zero, default is 0.001

    //--------------------------------------------------------------------------
    // Misc 
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

        _numThreads = 16;
        _algToUse = 0;
        _useClusters = true;
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
        _numThreads = 16;
        _algToUse = 0;
        _useClusters = true;
        setModelPath(modelpath);
        train(src, labels);
    }

    ~xLBPH() { }

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

    // See FaceRecognizer::load.
    void load(const FileStorage& fs);
    void load(const String &filename);
    void load();

    // See FaceRecognizer::save.
    void save(FileStorage& fs) const;
    
    CV_IMPL_PROPERTY(int, GridX, _grid_x)
    CV_IMPL_PROPERTY(int, GridY, _grid_y)
    CV_IMPL_PROPERTY(int, Radius, _radius)
    CV_IMPL_PROPERTY(int, Neighbors, _neighbors)
    CV_IMPL_PROPERTY(double, Threshold, _threshold)
   
    // path getters/setters
    void setModelPath(String modelpath);
    String getModelPath() const;
    String getModelName() const;
    String getInfoFile() const;
    String getHistogramsDir() const;
    String getHistogramFile(int label) const;
    String getHistogramAveragesFile() const;

    void setAlgToUse(int alg);
    void setNumThreads(int numThreads);

    //--------------------------------------------------------------------------
    // Additional Public Functions 
    // NOTE: Remember to add header to opencv2/face/facerec.hpp
    //--------------------------------------------------------------------------
    
    void setMCLSettings(int numIters, int e, double r);
    void setClusterSettings(double tierStep, int numTiers, int maxIters);

    void setUseClusters(bool flag);
    void test();

};

void xLBPH::setUseClusters(bool flag) {
    _useClusters = flag;
}


void xLBPH::setClusterSettings(double tierStep, int numTiers, int maxIters) {
    cluster_tierStep = tierStep;
    cluster_numTiers = numTiers;
    cluster_max_iterations = maxIters;
} 


void xLBPH::setMCLSettings(int numIters, int e, double r) {
    mcl_iterations = numIters;
    mcl_expansion_power = e;
    mcl_inflation_power = r;
}

int xLBPH::getMaxThreads() const {
    return _numThreads; 
}

int xLBPH::getLabelThreads() const {
    int threads = (int)floor(sqrt(_numThreads));
    return threads <= 0 ? 1 : threads;
}

int xLBPH::getHistThreads() const {
    int threads = (int)ceil(sqrt(_numThreads));
    return threads <= 0 ? 1 : threads;
}

void xLBPH::setNumThreads(int numThreads) {
    _numThreads = numThreads; 
}


void xLBPH::setAlgToUse(int alg) {
    _algToUse = alg; 
}


//------------------------------------------------------------------------------
// Model Path and Model File Getters/Setters 
//------------------------------------------------------------------------------

// Sets _modelpath, extracts model name from path, and sets _modelname

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

String xLBPH::getModelPath() const {
    return _modelpath; 
}

String xLBPH::getModelName() const {
    return _modelname;
} 

String xLBPH::getInfoFile() const {
    return getModelPath() + "/" + getModelName() + ".yml";
}

String xLBPH::getHistogramsDir() const {
    return getModelPath() + "/" + getModelName() + "-histograms";
}

String xLBPH::getHistogramFile(int label) const {
    char labelstr[16];
    sprintf(labelstr, "%d", label);
    return getHistogramsDir() + "/" + getModelName() + "-" + labelstr + ".bin";
}

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
    
    _model.test();
    printf("\n\n");

    String goodpath = "/dd-data/models/xlbph-test";
    ModelStorage goodmodel(goodpath, 1, 8, 12, 12);

    String badpath = "/dd-data/models/xlbph-test-bad-model";
    ModelStorage badmodel(badpath, 1, 8, 12, 12);

    String badpath2 = "/dd-data/videos/aws-test";
    ModelStorage badmodel2(badpath2, 1, 8, 12, 12);

    printf(" -==## Testing Model Storage Member Functions ##==- \n");
    printf("== Testing Model Information Functions == \n");
    printf(" - getModelPath\n");
    checkStr(_modelpath, _model.getPath());
    checkStr(goodpath, goodmodel.getPath());
    checkStr(badpath, badmodel.getPath());
    checkStr(badpath2, badmodel2.getPath());
    printf("\n");
    
    printf(" - getModelName\n");
    checkStr(_modelname, _model.getName());
    checkStr("xlbph-test", goodmodel.getName());
    checkStr("xlbph-test-bad-model", badmodel.getName());
    checkStr("aws-test", badmodel2.getName());
    printf("\n");

    printf(" - modelExists\n");
    checkBool(true, _model.exists());
    checkBool(true, goodmodel.exists());
    checkBool(false, badmodel.exists());
    checkBool(true, badmodel2.exists());
    printf("\n");

    printf(" - isValidModel\n");
    checkBool(true, _model.isValidModel());
    checkBool(true, goodmodel.isValidModel());
    checkBool(false, badmodel.isValidModel());
    checkBool(false, badmodel2.isValidModel());
    printf("\n");

    printf("== Testing Model File Getters (new) Functions == \n");
    printf(" - getLabelHistogramsFile()\n");
    checkStr(_model.getPath() + "/" + _model.getName() + "-labels/" + _model.getName() + "-label-12/" + _model.getName() + "-label-12-histograms.bin", _model.getLabelHistogramsFile(12));
    checkStr("/dd-data/models/xlbph-test/xlbph-test-labels/xlbph-test-label-12/xlbph-test-label-12-histograms.bin", goodmodel.getLabelHistogramsFile(12));
    printf("\n");

    printf("== Testing Model Creation/Manipulation Functions == \n");
    String testpath = "/dd-data/models/xlbph-model-storage-test";
    ModelStorage testmodel(testpath, -1, -1, -1, -1);

    printf(" - create\n");
    testmodel.create();

    AlgSettings testalg = {1, 8, 12, 12};

    std::map<int, int> testlabelinfo;
    testlabelinfo[1] = 11;
    testlabelinfo[2] = 22;
    testlabelinfo[3] = 33;
    testlabelinfo[4] = 44;
    testlabelinfo[5] = 55;

    printf(" - getAlgSettings - pre write/load\n");
    AlgSettings algPreLoad = testmodel.getAlgSettings();
    printf("algPreLoad.: {%d, %d, %d, %d}\n", algPreLoad.radius, algPreLoad.neighbors, algPreLoad.grid_x, algPreLoad.grid_y);
    printf("\n");

    printf(" - writeMetadata\n");
    testmodel.writeMetadata(testalg, testlabelinfo);
    

    printf(" - loadMetadata\n");
    AlgSettings algLoad;
    std::map<int,int> testlabelinfoLoad;
    testmodel.loadMetadata(algLoad, testlabelinfoLoad);
    printf("algLoad.: {%d, %d, %d, %d}\n", algLoad.radius, algLoad.neighbors, algLoad.grid_x, algLoad.grid_y);
    printf("testlabelinfoLoad:\n");
    for(std::map<int,int>::const_iterator it = testlabelinfoLoad.begin(); it != testlabelinfoLoad.end(); it++) {
        printf("  [%d] = %d\n", it->first, it->second);
    }
    printf("\n");


    printf(" - getAlgSettings - post write/load\n");
    AlgSettings algPostLoad = testmodel.getAlgSettings();
    printf("algPostLoad.: {%d, %d, %d, %d}\n", algPostLoad.radius, algPostLoad.neighbors, algPostLoad.grid_x, algPostLoad.grid_y);
    printf("\n");

    
    

    printf("\n");
    printf(" !! End of Member Functions Tests !!\n");
    printf("\n");
    
    printf("\n");
    printf("\n");
}


bool xLBPH::matsEqual(const Mat &a, const Mat &b) const {
    return countNonZero(a!=b) == 0; 
}

int xLBPH::getHistogramSize() const {
    return (int)(std::pow(2.0, static_cast<double>(_neighbors)) * _grid_x * _grid_y);
}

bool xLBPH::exists(const String &filepath) const {
    struct stat buffer;   
    return (stat (filepath.c_str(), &buffer) == 0);   
}


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

void xLBPH::calcHistogramAverages_thread(const std::vector<int> &labels, std::vector<Mat> &avgsdst) const {
    for(size_t idx = 0; idx < labels.size(); idx++) {
        Mat histavg;
        averageHistograms(_histograms.at(labels.at(idx)), histavg);
        avgsdst.push_back(histavg);
    } 
}

bool xLBPH::calcHistogramAverages() const {
        
    /*
    std::vector<Mat> averages;
    std::vector<int> labels; 
    for(std::map<int,int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); it++)
        labels.push_back(it->first);
    
    performMultithreadedCalc<int, Mat>(labels, averages, getMaxThreads(), &xLBPH::calcHistogramAverages_thread);

    return writeHistograms(getHistogramAveragesFile(), averages, false);
    */

    tbb::concurrent_vector<std::pair<int, Mat>> concurrent_averages;

    tbb::parallel_for_each(_histograms.begin(), _histograms.end(),
        [&concurrent_averages, this](std::pair<int, std::vector<Mat>> it) {
           Mat histavg;
           this->averageHistograms(it.second, histavg);

           concurrent_averages.push_back(std::pair<int, Mat>(it.first, histavg));
        } 
    );

    std::vector<Mat> averages;
    for(std::map<int,int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); it++) {
        int label = it->first; 

        //find it in concurrent_averages
        for(tbb::concurrent_vector<std::pair<int, Mat>>::const_iterator avg = concurrent_averages.begin(); avg != concurrent_averages.end(); avg++) {
            if(avg->first == label) {
                averages.push_back(avg->second);
                break;
            }
        }
    }

    return writeHistograms(getHistogramAveragesFile(), averages, false);
}

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
void xLBPH::clusterHistograms() {
    /* What is Histogram Clustering?
     * The idea is to group like histograms together
     * Every label has a set of clusters
     * Every cluster has an average histogram and a set of histograms
     */
    
    std::vector<int> labels;
    for(std::map<int, int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); it++)
        labels.push_back(it->first);

    std::vector<std::pair<int, std::vector<clstr::cluster_t>>> allClusters;

    int count = 0;
    tbb::parallel_for_each(_histograms.begin(), _histograms.end(), 
        [&](std::pair<int, std::vector<Mat>> it) {
        
        std::cout << "Clustering histograms " << count++ << " / " << (int)_histograms.size() << "                                \r" << std::flush;

        clstr::cluster_vars vars = {cluster_tierStep, 
                                    cluster_numTiers, 
                                    cluster_max_iterations,
                                    mcl_iterations, 
                                    mcl_expansion_power, 
                                    mcl_inflation_power, 
                                    mcl_prune_min};

        std::vector<clstr::cluster_t> labelClusters;
        clstr::clusterHistograms(_histograms[it.first], labelClusters, vars);

        printf("Found %d clusters for label %d\n", labelClusters.size(), it.first);

        //push all of the label clusters to the main clusters
        for(size_t i = 0; i < labelClusters.size(); i++) {
            _clusters[it.first].push_back(labelClusters.at((int)i));
        }

    });

    printf("Finished clustering histograms for %d labels                                        \n", (int)_histograms.size());

}



//------------------------------------------------------------------------------
// Standard Functions and File IO
//------------------------------------------------------------------------------
void xLBPH::load() {
     
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
}


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
void xLBPH::calculateHistograms(const std::vector<Mat> &src, std::vector<Mat> &dst) const {

    for(size_t idx = 0; idx < src.size(); idx++) {
        Mat lbp_image = elbp(src.at(idx), _radius, _neighbors);

        // get spatial histogram from this lbp image
        Mat p = spatial_histogram(
                lbp_image, // lbp_image
                static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), // number of possible patterns
                _grid_x, // grid size x
                _grid_y, // grid size y
                true);
        
        dst.push_back(p);
    }
}

void xLBPH::calculateLabels(const std::vector<std::pair<int, std::vector<Mat>>> &labelImages, std::vector<std::pair<int, int>> &labelinfo) const {
    
    for(size_t idx = 0; idx < labelImages.size(); idx++) {
        std::cout << "Calculating histograms " << (int)idx << " / " << (int)labelImages.size() << "          \r" << std::flush;
        int label = labelImages.at((int)idx).first;
        std::vector<Mat> imgs = labelImages.at((int)idx).second;
        
        std::vector<Mat> hists;
        performMultithreadedCalc<Mat, Mat>(imgs, hists, getHistThreads(), &xLBPH::calculateHistograms);

        labelinfo.push_back(std::pair<int, int>(label, (int)hists.size()));
        
        writeHistograms(getHistogramFile(label), hists, false);
    }
    std::cout << "\nOne thread done\n";
}


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
    
    std::vector<int> uniqueLabels;
    std::vector<int> numhists;

    // start training
    if(preserveData)
    {
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
                writeHistograms(getHistogramFile(it.first), hists, true);
                hists.clear();
            }
        );
        /*
        for(std::map<int, std::vector<Mat>>::const_iterator it = labelImages.begin(); it != labelImages.end(); ++it) {
            std::cout << "Calculating histograms for label " << labelcount << " / " << labelImages.size() << " [" << it->first << "]\r" << std::flush;

            //label = it->first;
            std::vector<Mat> imgs = it->second;
            std::vector<Mat> hists;
          
            
            performMultithreadedCalc<Mat, Mat>(imgs, hists, getMaxThreads(), &xLBPH::calculateHistograms);

            uniqueLabels.push_back(it->first);
            numhists.push_back((int)imgs.size());
            writeHistograms(getHistogramFile(it->first), hists, preserveData);
            
            hists.clear();

            labelcount++;
        }
        */
    }
    else {
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
                writeHistograms(getHistogramFile(label), hists, false);
            }
        );

        std::vector<std::pair<int, int>> labelInfoVec(concurrent_labelInfoVec.begin(), concurrent_labelInfoVec.end());
        for(size_t idx = 0; idx < labelInfoVec.size(); idx++) {
            uniqueLabels.push_back(labelInfoVec.at((int)idx).first);
            numhists.push_back(labelInfoVec.at((int)idx).second);
        }
    }
    
    std::cout << "Finished calculating histograms for " << labelImages.size() << " labels.                      \n";

    std::cout << "Writing infofile\n";
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

    // lastly we need to set _labelinfo
    _labelinfo = std::map<int, int>(); // if _labelinfo was set then clear it
    for(size_t labelIdx = 0; labelIdx < uniqueLabels.size(); labelIdx++) {
        _labelinfo[uniqueLabels.at((int)labelIdx)] = numhists.at((int)labelIdx);
    }
    mmapHistograms();

    std::cout << "Calculating histogram averages...\n";
    calcHistogramAverages();
    mmapHistogramAverages();    

    if(_useClusters)
        clusterHistograms();

    //load();

    std::cout << "Training complete\n";
}

//------------------------------------------------------------------------------
// Prediction Functions 
//------------------------------------------------------------------------------

void xLBPH::compareHistograms(const Mat &query, const std::vector<Mat> &hists, std::vector<double> &dists) const {
    for(size_t idx = 0; idx < hists.size(); idx++) {
        dists.push_back(compareHist(hists.at((int)idx), query, COMP_ALG));
    } 
}

// compares a given set of labels against the given query, each label is represented as a std::pair<int, std::vector<Mat>>
// where the int is the label and the std::vector<Mat> is that label's histograms
// TODO: we don't need to provide the histograms for each label, we can just give the label
// and get the histograms from our _histograms member
// See new compareLabelWithQuery below this func
void xLBPH::compareLabelHistograms(const Mat &query, const std::vector<std::pair<int, std::vector<Mat>>> &labelhists, std::vector<std::pair<int, std::vector<double>>> &labeldists) const {

    for(size_t idx = 0; idx < labelhists.size(); idx++) {
        int label = labelhists.at((int)idx).first;
        std::vector<Mat> hists = labelhists.at((int)idx).second;
        std::vector<double> dists;
        performMultithreadedComp<Mat, Mat, double>(query, hists, dists, getHistThreads(), &xLBPH::compareHistograms);
        std::sort(dists.begin(), dists.end());

        labeldists.push_back(std::pair<int, std::vector<double>>(label, dists));
    }
}

// compares a given set of labels against the given query
// each label is given as just it's integer value, histograms are gathered from _histograms member
// each label gets a vector of distances as doubles, in the same order as the histograms are in _histograms
// these distance vectors are compiled into one vector of vectors
/*
void xLBPH::compareLabelWithQuery(const Mat &query, const std::vector<int> &labels, std::vector<std::vector<double>> &labeldists) const {
    
    //std::map<int, std::vector<Mat>> _histograms;
    for(size_t idx = 0; idx < labels.size(); idx++) {
        std::vector<double> dists;
        performMultithreadedComp<Mat, Mat, double>(query, _hitograms.at((int)idx), dists, getHistThreads(), &xLBPH::compareHistograms);
        std::sort(dists.begin(), dists.end());
        labeldists.push_back(dists);
    }
}
*/

//void xLBPH::predict_avg_clustering(InputArray _query, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::set<int> &labels) const {
void xLBPH::predict_avg_clustering(InputArray _query, tbb::concurrent_vector<std::pair<double, int>> &bestpreds, const std::vector<int> &labels) const {
//void xLBPH::predict_avg_clustering(InputArray _query, int &minClass, double &minDist) const {

    if(!_useClusters) {
        CV_Error(Error::StsError, "Cannot make prediction using clusters, clustering disabled!"); 
        return;
    }

    Mat query = _query.getMat();
    
    tbb::concurrent_vector<std::pair<double, int>> bestlabels;
    /*
    tbb::parallel_for_each(_histavgs.begin(), _histavgs.end(), 
        [&bestlabels, &query, &labels](std::pair<int, Mat> it) {
            if(labels.find(it.first) != labels.end()) {
                bestlabels.push_back(std::pair<double, int>(compareHist(it.second, query, COMP_ALG), it.first));
            }
        }
    );
    */

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


    // find best cluster for each best label
    /*
    std::vector<std::pair<int, std::vector<Mat>>> labelhists;
    for(size_t idx = 0; idx < bestlabels.size() && (int)idx < numLabelsToCheck; idx++) {
        int label = bestlabels.at(idx).second;
        //printf("Finding best cluster for PID %d\n", label);
        
        std::vector<std::pair<Mat, std::vector<Mat>>> labelClusters = _clusters.at(label);
        std::vector<Mat> clusterAvgs;
        for(size_t clusterIdx = 0; clusterIdx < labelClusters.size(); clusterIdx++) {
            clusterAvgs.push_back(labelClusters.at(clusterIdx).first);
        }
        
        //printf(" - Has %d clusters, dispatching comparison threads...\n", (int)clusterAvgs.size());
        std::vector<double> clusterAvgsDists;
        performMultithreadedComp<Mat, Mat, double>(query, clusterAvgs, clusterAvgsDists, getHistThreads(), &xLBPH::compareHistograms);
      
        //printf(" - Finding best cluster...");
        std::vector<std::pair<double, int>> bestClusters; 
        for(size_t clusterIdx = 0; clusterIdx < clusterAvgsDists.size(); clusterIdx++) {
            bestClusters.push_back(std::pair<double, int>(clusterAvgsDists.at((int)clusterIdx), (int)clusterIdx));
        }
        std::sort(bestClusters.begin(), bestClusters.end());
        

        // figure out how many clusters to check per label
        // TODO: calculate this on a per label basis
        int numClustersToCheck = 2;
       
        //printf(" - Using top %d best clusters...\n", numClustersToCheck);
        std::vector<Mat> combinedClusters;
        for(size_t bestIdx = 0; bestIdx < bestClusters.size() && (int)bestIdx < numClustersToCheck; bestIdx++) {
            std::vector<Mat> cluster = labelClusters.at(bestClusters.at((int)bestIdx).second).second; 
            for(size_t clusterIdx = 0; clusterIdx < cluster.size(); clusterIdx++) {
               combinedClusters.push_back(cluster.at((int)clusterIdx));
            }
        }

        //printf(" - Pushing combined clusters to labelhists...\n");
        labelhists.push_back(std::pair<int, std::vector<Mat>>(label, combinedClusters));
    }
    */

    tbb::concurrent_vector<std::pair<int, std::vector<Mat>>> labelhists;
    tbb::parallel_for(0, numLabelsToCheck, 1, 
        [&](int i) {
            
            try {

                //printf(" - %d: bestlabels.at\n", i);
                int label = bestlabels.at(i).second;
                //printf(" - %d: label = %d | _clusters.at\n", i, label);
                std::vector<std::pair<Mat, std::vector<Mat>>> labelClusters = _clusters.at(label);
                tbb::concurrent_vector<std::pair<double, int>> clusterDists;
                
                //printf(" - %d: clusters part 1 parallel\n", i);
                tbb::parallel_for(0, (int)labelClusters.size(), 1,
                    [&i, &labelClusters, &clusterDists, &query](int clusterIdx) {
                        //printf(" - %d: clusterIdx = %d | labelClusters.at in clusterDists.push_back\n", i, clusterIdx);
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

                std::vector<Mat> combinedClusters;
                for(size_t bestIdx = 0; bestIdx < clusterDists.size() && (int)bestIdx < numClustersToCheck; bestIdx++) {
                    
                    //printf(" - %d: bestIdx = %d | clusterDists.at\n", i, (int)bestIdx);
                    int labelClustersIdx = clusterDists.at((int)bestIdx).second;

                    //printf(" - %d: bestIdx = %d | labelClustersIdx = %d | labelClusters.at\n", i, (int)bestIdx, labelClustersIdx);
                    std::vector<Mat> cluster = labelClusters.at(labelClustersIdx).second; 

                    for(size_t clusterIdx = 0; clusterIdx < cluster.size(); clusterIdx++) {
                        //printf(" - %d: bestIdx = %d | clusterIdx = %d | cluster.at\n", i, (int)bestIdx, (int)clusterIdx);
                       combinedClusters.push_back(cluster.at((int)clusterIdx));
                    }
                }

                //printf(" - Pushing combined clusters to labelhists...\n");
                labelhists.push_back(std::pair<int, std::vector<Mat>>(label, combinedClusters));

                //printf(" - %d: done\n", i);
            }            
            catch (const std::exception &e) {
                printf(" - %d: CAUGHT EXCEPTION | %s\n", i, e.what()); 
                std::exit(1);
            } 

        }
    );

    //printf(" - Calculating distances for best clusters...\n");
   
    // check best labels by cluster

    printf(" - organizing\n");
    tbb::concurrent_vector<std::pair<int, std::vector<double>>> labeldists;
    tbb::parallel_for_each(labelhists.begin(), labelhists.end(),
        [&labelhists, &labeldists, &query](std::pair<int, std::vector<Mat>> it) {

            std::vector<Mat> hists = it.second;
            tbb::concurrent_vector<double> dists;

            tbb::parallel_for_each(hists.begin(), hists.end(), 
                [&labeldists, &dists, &query](Mat hist) {
                    dists.push_back(compareHist(hist, query, COMP_ALG));
                } 
            );
            labeldists.push_back(std::pair<int, std::vector<double>>(it.first, std::vector<double>(dists.begin(), dists.end())));
        } 
    ); 

    /*
    printf(" - Dists found:\n");
    for(size_t idx = 0; idx < labeldists.size(); idx++) {
        int label = labeldists.at((int)idx).first;
        std::vector<double> dists = labeldists.at((int)idx).second;
        printf("    - %d: ", label);
        for(int i = 0 ; i < (int)dists.size(); i++) {
            printf("%0.3f, ", dists.at(i));
        }
        printf("\n");
    }
    */

    //printf(" - Grabbing best predictions for each PID...\n");
    //std::vector<std::pair<double, int>> bestpreds;

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
    /*
    tbb::parallel_for_each(_histavgs.begin(), _histavgs.end(), 
        [&bestlabels, &query, &labels](std::pair<int, Mat> it) {
            if(labels.find(it.first) != labels.end()) {
                bestlabels.push_back(std::pair<double, int>(compareHist(it.second, query, COMP_ALG), it.first));
            }
        }
    );
    */
    
    printf("Finding bestlabels...\n");
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

    printf("Checking %d best labels...\n", numLabelsToCheck);

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

    //minDist = DBL_MAX;
    //minClass = -1;
    /*
    tbb::parallel_for_each(_histograms.begin(), _histograms.end(),
        [&bestpreds, &query, &labels](std::pair<int, std::vector<Mat>> it) {
            if(labels.find(it.first) != labels.end()) {
                std::vector<Mat> hists = it.second;
                
                tbb::concurrent_vector<double> dists;
                tbb::parallel_for_each(hists.begin(), hists.end(), 
                    [&dists, &query](Mat hist) {
                        dists.push_back(compareHist(hist, query, COMP_ALG));
                    } 
                );

                std::sort(dists.begin(), dists.end());
                
                bestpreds.push_back(std::pair<double, int>(dists.at(0), it.first));
            }
        }
    );
    std::sort(bestpreds.begin(), bestpreds.end());
    */    

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
    
    printf("Extracting labels...\n");
    // Gets the list of labels to check
    Mat labelsMat = _labels.getMat();
    std::vector<int> labels;
    for(size_t labelIdx = 0; labelIdx < labelsMat.total(); labelIdx++)
        labels.push_back(labelsMat.at<int>((int)labelIdx));
    printf("Found %d labels...\n", (int)labels.size());


    printf("Calling prediction algorithm...\n");
    tbb::concurrent_vector<std::pair<double, int>> bestpreds;
    switch(_algToUse) {
        case 1: predict_avg(query, bestpreds, labels); break;
        case 2: predict_avg_clustering(query, bestpreds, labels); break;
        default: predict_std(query, bestpreds, labels); break;
    }
    
    
    printf("Compiling prediction results...\n");
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
