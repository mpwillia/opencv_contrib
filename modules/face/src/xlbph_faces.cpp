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
#include "opencv2/face.hpp"
#include "face_basic.hpp"
#include "mcl.hpp"

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

    // label info
    std::map<int, int> _labelinfo;

    // histograms
    std::map<int, std::vector<Mat>> _histograms;
    std::map<int, Mat> _histavgs;
    std::map<int, Mat> _distmats;    
    std::map<int, std::vector<std::pair<Mat, std::vector<Mat>>>> _clusters;

    // defines what prediction algorithm to use
    int _algToUse;
    


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
    void predict_std(InputArray _src, int &label, double &dist) const;
    void predict_avg(InputArray _src, int &label, double &dist) const;
    void predict_avg_clustering(InputArray _query, int &minClass, double &minDist) const;

    void compareLabelHistograms(const Mat &query, const std::vector<std::pair<int, std::vector<Mat>>> &labelhists, std::vector<std::pair<int, std::vector<double>>> &labeldists) const;
    //void compareLabelWithQuery(const Mat &query, const std::vector<int> &labels, std::vector<std::vector<double>> &labeldists) const;
    void compareHistograms(const Mat &query, const std::vector<Mat> &hists, std::vector<double> &dists) const;
    
    const int minLabelsToCheck = 5;
    const double labelsToCheckRatio = 0.05;

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
                _threshold(threshold) {

        _numThreads = 16;
        _algToUse = 0;
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
                _threshold(threshold) {
        _numThreads = 16;
        _algToUse = 0;
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
    void setClusterSettings(double tierStep, int numTiers);

    void test();
};

void xLBPH::setClusterSettings(double tierStep, int numTiers) {
    cluster_tierStep = tierStep;
    cluster_numTiers = numTiers;
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

void xLBPH::test() {
    // make some fake hists
    int numhists = 16;
    int size = 4;
    std::vector<Mat> histsToSave;
    
    std::cout << "Making test hists...\n";
    for(int i = 0; i < numhists - 2; i++) {
        Mat mat = Mat::zeros(1, size, CV_32FC1);
        mat += i;
        histsToSave.push_back(mat);
    }
    Mat matmax = Mat::zeros(1, size, CV_32FC1);
    Mat matmin = Mat::zeros(1, size, CV_32FC1);
    matmax = FLT_MAX;
    matmin = FLT_MIN;
    histsToSave.push_back(matmax);
    histsToSave.push_back(matmin);
  
    std::cout << "Saving test hists...\n";
    String testhistsfile = getHistogramsDir() + "/testhists.bin";
    // write them to a fake file
    FILE *writefp = fopen(testhistsfile.c_str(), "w");
    if(writefp == NULL) 
        CV_Error(Error::StsError, "Cannot open histogram file '"+testhistsfile+"'");

    float* writebuffer = new float[size * numhists];
    for(size_t sampleIdx = 0; (int)sampleIdx < numhists; sampleIdx++) {
        memcpy((writebuffer + sampleIdx * size), histsToSave.at((int)sampleIdx).ptr<float>(), size * sizeof(float));
    }
    fwrite(writebuffer, sizeof(float), size * numhists, writefp);
    delete writebuffer;
    fclose(writefp);
  

    std::cout << "Mapping test hists file...\n";
    // mmap fake file
    int fd = open(testhistsfile.c_str(), O_RDONLY);
    if(fd < 0)
        CV_Error(Error::StsError, "Cannot open histogram file '"+testhistsfile+"'");

    char* mapPtr = (char*)mmap(NULL, size * numhists * sizeof(float), PROT_READ, MAP_PRIVATE, fd, 0);
    if(mapPtr == MAP_FAILED)
        CV_Error(Error::StsError, "Cannot mem map file '"+testhistsfile+"'");
    
    std::cout << "Loading query mats...\n";
    // make matricies from map
    std::vector<Mat> query;
    for(int i = 0; i < numhists; i++) {
        Mat mat(1, size, CV_32FC1, mapPtr + (size * sizeof(float) * i));
        query.push_back(mat);
    }

    
    std::cout << "Loading check mat...\n";
    // load mats from file 
    std::vector<Mat> check;
    FILE *readfp = fopen(testhistsfile.c_str(), "r");
    if(readfp == NULL) 
        CV_Error(Error::StsError, "Cannot open histogram file '"+testhistsfile+"'");
    
    float readbuffer[size];
    while(fread(readbuffer, sizeof(float), size, readfp) > 0) {
        Mat hist = Mat::zeros(1, size, CV_32FC1);
        memcpy(hist.ptr<float>(), readbuffer, size * sizeof(float));
        check.push_back(hist);
    }
    fclose(readfp);

    std::cout << "Comparing results...\n";
    // compare results
    CV_Assert(query.size() == check.size());
    CV_Assert(query.size() == histsToSave.size());

    std::cout << "saved size: " << histsToSave.size() << "  |  query size: " << query.size() << "  |  check size: " << check.size() << "\n";
    for(size_t idx = 0; idx < query.size(); idx++) {
        
        std::cout << "idx: " << idx << std::flush;
        std::cout << "  |  saved: " << matToHex(histsToSave.at(idx)) << std::flush;
        std::cout << "  |  query: " << matToHex(query.at(idx)) << std::flush;
        std::cout << "  |  check: " << matToHex(check.at(idx)) << std::flush;
        std::cout << "\n";
        //std::cout << "saved: " << matToString(histsToSave.at(idx)) <<"  |  query: " << matToString(query.at(idx)) << "  |  check: " << matToString(check.at(idx)) << "\n";
        if(!matsEqual(query.at(idx), check.at(idx)))
        {
            //std::cout << "query: " << matToString(query.at(idx)) << "  |  " << matToString(check.at(idx)) << " :check" << "\n";
            CV_Error(Error::StsError, "MATS NOT EQUAL!!!");
        }
    }
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
    
    std::vector<Mat> averages;
    std::vector<int> labels; 
    for(std::map<int,int>::const_iterator it = _labelinfo.begin(); it != _labelinfo.end(); it++)
        labels.push_back(it->first);
    
    performMultithreadedCalc<int, Mat>(labels, averages, getMaxThreads(), &xLBPH::calcHistogramAverages_thread);

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

        //struct stat st;
        //stat(filename.c_str(), &st);
        //char* mapPtr = (char*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
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
// Calculates the weights between each histogram and puts them in weights
void xLBPH::cluster_calc_weights(Mat &dists, Mat &weights, double tierStep, int numTiers) {
    weights.create(dists.rows, dists.cols, dists.type());

    // calculate tiers and weights
    for(size_t i = 0; i < dists.rows; i++) {
        // find best
        double best = DBL_MAX;
        for(size_t j = 0; j < dists.cols; j++) {
            double check = dists.at<double>(j,i);
            if(check > 0 && check < best) 
                best = check;
        }
        
        // calculate tiers
        for(size_t j = 0; j < dists.cols; j++) {
            double check = dists.at<double>(j,i);
            if(check > 0 && check != best) 
                weights.at<double>(j,i) = ceil(((check - best) / best) / tierStep);
            else 
                weights.at<double>(j,i) = 1; 
        }

        // calculate weights
        for(size_t j = 0; j < dists.cols; j++) {
            double weight = (numTiers+1) - weights.at<double>(j,i);
            weights.at<double>(j,i) = (weight <= 0) ? 0 : weight;
        }
    }
}

// Finds clusters for the given label's dists and puts the MCL mat in mclmat
void xLBPH::cluster_dists(Mat &dists, Mat &mclmat, double r) {
    //printf("\t\t\t - clustering dists...\n");
    mclmat.create(dists.rows, dists.cols, dists.type());

    // find weights
    cluster_calc_weights(dists, mclmat, cluster_tierStep, cluster_numTiers);

    // iterate
    mcl::cluster(mclmat, mcl_iterations, mcl_expansion_power, r, mcl_prune_min);
}


// Interprets a given MCL matrix as clusters
void xLBPH::cluster_interpret(Mat &mclmat, std::vector<std::set<int>> &clusters) {
    //printf("\t\t\t - interpreting clusters...\n");
    // interpret clusters
    std::map<int, std::set<int>> clusters_map;
    for(int vert = 0; vert < mclmat.rows; vert++) {
        //std::cout << "checking vert " << vert << "\n";
        for(int check = 0; check < mclmat.cols; check ++) {
            double dist = mclmat.at<double>(check, vert);
            //std::cout << "\tchecking check " << check << " | dist: " << dist << "\n";
            if(dist > 0) {
                // we want to add it
                // check if it already has been added somewhere
                bool found = false;
                for(int i = 0; i < vert; i++) {
                    if(!clusters_map[i].empty() && clusters_map[i].find(vert) != clusters_map[i].end()) {
                        //std::cout << "\t\tfound check at " << i << " - not adding\n";
                        found = true; 
                    } 
                }
                    
                if(!found) {
                    //std::cout << "\t\tdidn't find check adding\n";
                    clusters_map[vert].insert(check);
                }
            }
        } 
    }
    
    for(std::map<int, std::set<int>>::const_iterator it = clusters_map.begin(); it != clusters_map.end(); it++) {
        if(!it->second.empty())
            clusters.push_back(it->second);
    }
}

double xLBPH::cluster_ratio(std::vector<std::set<int>> &clusters) {
    int numHists = 0;
    int worstCase = 0;
    for(size_t idx = 0; idx < clusters.size(); idx++) {
        std::set<int> cluster = clusters.at(idx);
        numHists += (int)cluster.size();
        if((int)cluster.size() > worstCase)
            worstCase = (int)cluster.size();
    }
    worstCase += (int)clusters.size();
    return  worstCase / (double)numHists;
}

void xLBPH::cluster_find_optimal(Mat &dists, std::vector<std::set<int>> &clusters) {
    

    //printf("\t - finding optimal cluster...\n");
    //printf("=========\n");
    int optimalClustersMax = ceil(sqrt(dists.rows));
    int optimalClustersMin = floor(sqrt(dists.rows));
    int optimalCase = (int)ceil(sqrt((int)dists.rows)*2);
    double optimalRatio = optimalCase / (double)dists.rows;
    /*
    printf("Optimal Case Checks: %d\n", optimalCase);
    printf("Optimal Check Ratio: %.3f\n", optimalRatio);
    printf("Optimal Clusters: %d - %d\n\n", optimalClustersMin, optimalClustersMax);
    */

    Mat initial;
    double r = mcl_inflation_power;

    printf("\t\t - initial r of %0.3f\n", r);

    cluster_dists(dists, initial, r);
    cluster_interpret(initial, clusters);
    
    int checkClusters = (int)clusters.size();
    int iterations = 5;
    int base = 7;
    bool makeLarger = (checkClusters < optimalClustersMin);
    for(int i = 0; i < iterations; i++) {
        if(checkClusters < optimalClustersMin && makeLarger)
            r *= (base + 1.0 + i) / base; // need more clusters - larger r
        else if(checkClusters > optimalClustersMax && !makeLarger) 
            r *= (base - 1.0 - i) / base; // need fewer clusters - smaller r
        else
            break;

        if(r <= 1)
            break;
    

        Mat mclmat;
        cluster_dists(dists, mclmat, r);
        clusters.clear();
        cluster_interpret(mclmat, clusters);
        checkClusters = (int)clusters.size();
    }



    //int prevClusters = optimalClustersMax;

    //printf("pre -> r: %0.3f  |  checkClusters: %d  |  prevClusters: %d  |  optimalClusters: %d - %d\n", r, checkClusters, prevClusters, optimalClustersMin, optimalClustersMax);
    /*
    double mcl_inflation_min = mcl_inflation_power / 2;
    if(mcl_inflation_min <= 1)
        mcl_inflation_min = 1.1;
    double mcl_inflation_max = mcl_inflation_power * 2;
    bool larger = true;
    while(checkClusters != prevClusters && checkClusters != optimalClustersMin && checkClusters != optimalClustersMax) {
        prevClusters = checkClusters;
        if(checkClusters < optimalClustersMin) {
            //printf("larger r: %.3f -> ", r);
            larger = true;
            // we want more clusters - larger r 
            if(r < mcl_inflation_power) // r <= baseline
                r = (r + mcl_inflation_power) / 2;
            else // r > baseline
                r = (r + mcl_inflation_max) / 2;
            //printf("%.3f\n", r);
        } 
        else if(checkClusters > optimalClustersMax) {
            //printf("smaller r: %.3f -> ", r);
            larger = false;
            // we want fewer clusters - smaller r
            if(r <= mcl_inflation_power) // r <= baseline | (1.5 + 2) / 2
                r = (r + mcl_inflation_min) / 2;
            else // r > baseline | (2.5 + 4) / 2
                r = (r + mcl_inflation_power) / 2;
            //printf("%.3f\n", r);
        }
        
        printf("\t\t - trying r of %0.3f\n", r);

        Mat mclmat;
        cluster_dists(dists, mclmat, r);
        clusters.clear();
        cluster_interpret(mclmat, clusters);
        checkClusters = (int)clusters.size();

        //printf("r: %0.3f  |  checkClusters: %d  |  prevClusters: %d  |  optimalClusters: %d - %d\n", r, checkClusters, prevClusters, optimalClustersMin, optimalClustersMax);
    }
    
    if(checkClusters != optimalClustersMin && checkClusters != optimalClustersMax) {
        if((larger && r != mcl_inflation_max) || (!larger && r != mcl_inflation_min))
        {
            if(larger) 
                r = mcl_inflation_max;
            else
                r = mcl_inflation_min;

            Mat mclmat;
             
            printf("\t\t - last chance r of %0.3f\n", r);
            cluster_dists(dists, mclmat, r);
            clusters.clear();
            cluster_interpret(mclmat, clusters);
            checkClusters = (int)clusters.size();

            //printf("last chance r: %0.3f  |  checkClusters: %d  |  optimalClusters: %d - %d\n", r, checkClusters, optimalClustersMin, optimalClustersMax);
        }
    }
    */    

}

void xLBPH::cluster_label(int label, std::vector<std::pair<Mat, std::vector<Mat>>> &matClusters) {
//void xLBPH::cluster_label(int label, std::vector<std::set<int>> &clusters) {
    
    std::vector<Mat> hists = _histograms[label];

    printf(" - calculating clusters for %d with %d histograms...\n", label, (int)hists.size());
    //std::cout << " - calculating clusters for " << label << " with " << (int)hists.size() << " histograms...\r" << std::flush;
    
    //performMultithreadedComp<Mat, Mat, double>(query, histavgs, avgdists, getMaxThreads(), &xLBPH::compareHistograms);
    /*
    Mat dists = Mat::zeros((int)hists.size(), (int)hists.size(), CV_64FC1);
    for(size_t i = 0; i < hists.size()-1; i++) {
        
    }
    */

    Mat dists = Mat::zeros((int)hists.size(), (int)hists.size(), CV_64FC1);
    // get raw dists
    for(size_t i = 0; i < hists.size()-1; i++) {
        for(size_t j = i; j < hists.size(); j++) {
            double dist = compareHist(hists.at((int)i), hists.at((int)j), COMP_ALG);
            dists.at<double>(i, j) = dist;
            dists.at<double>(j, i) = dist;
        } 
    }

    std::vector<std::set<int>> clusters;
    cluster_find_optimal(dists, clusters);
     
    printf("\t - %d has %d clusters for %d histograms -> averaging clusters...\n", label, (int)clusters.size(), (int)hists.size());
    //std::cout << " - " << label << " has " << (int)clusters.size() << " clusters for " << (int)hists.size() << " histograms -> averaging clusters...\r" << std::flush;

    //std::vector<std::pair<Mat, std::vector<Mat>>> matClusters;
    for(size_t i = 0; i < clusters.size(); i++) {
        std::set<int> cluster = clusters.at((int)i);
        
        std::vector<Mat> clusterHists;
        Mat clusterAvg;

        for(std::set<int>::const_iterator it = cluster.begin(); it != cluster.end(); it++) {
            clusterHists.push_back(hists.at(*it));
        }
        
        averageHistograms(clusterHists, clusterAvg);

        matClusters.push_back(std::pair<Mat, std::vector<Mat>>(clusterAvg, clusterHists));
    }
    
    printf("\t - finished with %d who has %d clusters for %d histograms \n", label, (int)clusters.size(), (int)hists.size());

    //void xLBPH::averageHistograms(const std::vector<Mat> &hists, Mat &histavg) const {
}

void xLBPH::clusterHistograms() {
    /* What is Histogram Clustering?
     * The idea is to group like histograms together
     * Every label has a set of clusters
     * Every cluster has an average histogram and a set of histograms
     */
    
    //double avgCheckRatio = 0;
    for(std::map<int, std::vector<Mat>>::const_iterator it = _histograms.begin(); it != _histograms.end(); it++) {
        
        int numHists = (int)it->second.size();
        std::vector<std::pair<Mat, std::vector<Mat>>> labelClusters;
        cluster_label(it->first, labelClusters);

        //push all of the label clusters to the main clusters
        for(size_t i = 0; i < labelClusters.size(); i++) {
            _clusters[it->first].push_back(labelClusters.at((int)i));
        }

        //std::map<int, std::vector<std::pair<Mat, std::vector<Mat>>>> _clusters;
        /*
        //double clusterRatio = (int)clusters.size() / (double)hists.size(); 
        int worstCase = 0;
        for(size_t idx = 0; idx < clusters.size(); idx++) {
            std::set<int> cluster = clusters.at(idx);
            if((int)cluster.size() > worstCase)
                worstCase = (int)cluster.size();
        }
        worstCase += (int)clusters.size();
        double checkRatio = worstCase / (double)numHists;
        avgCheckRatio += checkRatio;
        printf("=== Cluster Stats [%d] ===\n", it->first);
        printf("Total Hists: - - > %7d\n", numHists);
        printf("Total Clusters:  > %7d\n", (int)clusters.size());
        printf("Worst Case Checks: %7d\n", worstCase);
        printf("Check Ratio: - - > %7.3f (lower = better)\n", checkRatio);
        //printf("%d Clusters from %d hists for %d - Cluster Ratio: %7.3f - Worst Case Checks: %d - Check Ratio: %7.3f\n", (int)clusters.size(), (int)hists.size(), it->first, ratio, worstCase, checkRatio);
        for(size_t idx = 0; idx < clusters.size(); idx++) {
            std::set<int> cluster = clusters.at(idx);
            for(std::set<int>::const_iterator it = cluster.begin(); it != cluster.end(); it++) {
                printf("%d, ", *it);
            }
            printf("\n");
        }
        printf("\n");
    
        //break;
        */
    }
    /*
    avgCheckRatio /= (int)_histograms.size();
    printf("\n### Overall ###\n");
    printf("Average Check Ratio: %7.3f\n", avgCheckRatio);
    */
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
        int labelcount = 0;
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
    }
    else {
        std::cout << "Multithreaded label calcs\n";
        std::vector<std::pair<int, std::vector<Mat>>> labelImagesVec(labelImages.begin(), labelImages.end());
        std::vector<std::pair<int, int>> labelInfoVec;
        //void xLBPH::calculateLabels(const std::vector<std::pair<int, std::vector<Mat>>> &labelImages, std::vector<std::pair<int, int>> &labelinfo) const {
        performMultithreadedCalc<std::pair<int, std::vector<Mat>>, std::pair<int, int>>(labelImagesVec, labelInfoVec, getLabelThreads(), &xLBPH::calculateLabels);

        for(size_t idx = 0; idx < labelInfoVec.size(); idx++) {
            uniqueLabels.push_back(labelInfoVec.at((int)idx).first);
            numhists.push_back(labelInfoVec.at((int)idx).second);
        }
    }
    
    std::cout << "Finished calculating histograms for " << labelImages.size() << " labels.            \n";

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

    std::cout << "Clustering Histograms...\n";
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

void xLBPH::predict_avg_clustering(InputArray _query, int &minClass, double &minDist) const {
    Mat query = _query.getMat();
    
    // we need to break histavgs into it's comps 
    std::vector<Mat> histavgs;
    std::vector<int> labels;
    for(std::map<int, Mat>::const_iterator it = _histavgs.begin(); it != _histavgs.end(); it++) {
        labels.push_back(it->first);
        histavgs.push_back(it->second);
    }

    // perform histogram comparison to find dists from query.
    std::vector<double> avgdists;
    performMultithreadedComp<Mat, Mat, double>(query, histavgs, avgdists, getMaxThreads(), &xLBPH::compareHistograms);
    
    // reassociate each dist with it's label
    std::vector<std::pair<double, int>> bestlabels;
    for(size_t idx = 0; idx < avgdists.size(); idx++) {
        bestlabels.push_back(std::pair<double, int>(avgdists.at((int)idx), labels.at((int)idx)));
    }

    // sort the data by smallest distance first
    std::sort(bestlabels.begin(), bestlabels.end());
    
    // figure out how many labels to check
    int numLabelsToCheck = (int)((int)_labelinfo.size() * labelsToCheckRatio);
    if(numLabelsToCheck < minLabelsToCheck)
        numLabelsToCheck = minLabelsToCheck;


    // find best cluster for each best label
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
      
        /*
        printf(" - Got %d dists back from threads -> ", (int)clusterAvgsDists.size());
        for(int i = 0; i < (int)clusterAvgsDists.size(); i++)
            printf("[%d: %0.3f], ", i, clusterAvgsDists.at(i));
        printf("\n");
        */

        //printf(" - Finding best cluster...");
        std::vector<std::pair<double, int>> bestClusters; 
        for(size_t clusterIdx = 0; clusterIdx < clusterAvgsDists.size(); clusterIdx++) {
            bestClusters.push_back(std::pair<double, int>(clusterAvgsDists.at((int)clusterIdx), (int)clusterIdx));
        }
        std::sort(bestClusters.begin(), bestClusters.end());
        
        /*
        printf(" - best clusters: ");
        for(int i = 0; i < (int)bestClusters.size(); i++) {
            std::pair<double, int> cluster = bestClusters.at(i);
            printf("[%d: %0.3f], ", cluster.second, cluster.first);
        }
        printf("\n");
        */

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
    
    //printf(" - Calculating distances for best clusters...\n");
    std::vector<std::pair<int, std::vector<double>>> labeldists;

    performMultithreadedComp<Mat, std::pair<int, std::vector<Mat>>, std::pair<int, std::vector<double>>>(query, labelhists, labeldists, getLabelThreads(), &xLBPH::compareLabelHistograms);
    
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
    std::vector<std::pair<double, int>> bestpreds;
    for(size_t idx = 0; idx < labeldists.size(); idx++) {
        std::vector<double> dists = labeldists.at((int)idx).second;
        bestpreds.push_back(std::pair<double, int>(dists.at(0), labeldists.at((int)idx).first));
    }
    std::sort(bestpreds.begin(), bestpreds.end());

    minDist = bestpreds.at(0).first;
    minClass = bestpreds.at(0).second;
    
    /*
    std::cout << "\nBest Prediction by PID:\n";
    for(std::vector<std::pair<double, int>>::const_iterator it = bestpreds.begin(); it != bestpreds.end(); ++it) {
        printf("[%d, %f]\n", it->first, it->second);
    }
    */
} 

void xLBPH::predict_avg(InputArray _query, int &minClass, double &minDist) const {
    Mat query = _query.getMat();

    //std::map<int, Mat> histavgs = _histavgs;

    // we need to break histavgs into it's comps 
    std::vector<Mat> histavgs;
    std::vector<int> labels;
    for(std::map<int, Mat>::const_iterator it = _histavgs.begin(); it != _histavgs.end(); it++) {
        labels.push_back(it->first);
        histavgs.push_back(it->second);
    }

    // perform histogram comparison to find dists from query.
    std::vector<double> avgdists;
    performMultithreadedComp<Mat, Mat, double>(query, histavgs, avgdists, getMaxThreads(), &xLBPH::compareHistograms);
    
    // reassociate each dist with it's label
    std::vector<std::pair<double, int>> bestlabels;
    for(size_t idx = 0; idx < avgdists.size(); idx++) {
        bestlabels.push_back(std::pair<double, int>(avgdists.at((int)idx), labels.at((int)idx)));
    }
    // sort the data by smallest distance first
    std::sort(bestlabels.begin(), bestlabels.end());
    
    // figure out how many labels to check
    int numLabelsToCheck = (int)((int)_labelinfo.size() * labelsToCheckRatio);
    if(numLabelsToCheck < minLabelsToCheck)
        numLabelsToCheck = minLabelsToCheck;
    
    // setup data for multithreading
    std::vector<std::pair<int, std::vector<Mat>>> labelhists;
    for(size_t idx = 0; idx < bestlabels.size() && (int)idx < numLabelsToCheck; idx++) {
        int label = bestlabels.at(idx).second;
        labelhists.push_back(std::pair<int, std::vector<Mat>>(label, _histograms.at(label)));
    }
    std::vector<std::pair<int, std::vector<double>>> labeldists;

    // perform histogram comparison on all the histograms of the top <numLabelsToCheck> best averages
    performMultithreadedComp<Mat, std::pair<int, std::vector<Mat>>, std::pair<int, std::vector<double>>>(query, labelhists, labeldists, getLabelThreads(), &xLBPH::compareLabelHistograms);
    
    // find best prediction for each label checked
    std::vector<std::pair<double, int>> bestpreds;
    for(size_t idx = 0; idx < labeldists.size(); idx++) {
        std::vector<double> dists = labeldists.at((int)idx).second;
        bestpreds.push_back(std::pair<double, int>(dists.at(0), labeldists.at((int)idx).first));
    }
    std::sort(bestpreds.begin(), bestpreds.end());

    minDist = bestpreds.at(0).first;
    minClass = bestpreds.at(0).second;
    
    /*
    std::cout << "\nBest Prediction by PID:\n";
    for(std::vector<std::pair<double, int>>::const_iterator it = bestpreds.begin(); it != bestpreds.end(); ++it) {
        printf("[%d, %f]\n", it->first, it->second);
    }
    */
} 


void xLBPH::predict_std(InputArray _query, int &minClass, double &minDist) const {
    Mat query = _query.getMat();

    //minDist = DBL_MAX;
    //minClass = -1;
    //std::map<int, double> bestpreds;
    std::vector<std::pair<double, int>> bestpreds;
    //void performMultithreadedComp(const S &query, const std::vector<S> &src, std::vector<D> &dst, int numThreads, void (xLBPH::*compFunc)(const S &query, const std::vector<S> &src, std::vector<D> &dst) const) const
    for(std::map<int, std::vector<Mat>>::const_iterator it = _histograms.begin(); it != _histograms.end(); ++it) {
        
        //bestpreds[it->first] = DBL_MAX;
        std::vector<Mat> hists = it->second;
        std::vector<double> dists;
        performMultithreadedComp<Mat, Mat, double>(query, hists, dists, getMaxThreads(), &xLBPH::compareHistograms);
        std::sort(dists.begin(), dists.end());
        
        bestpreds.push_back(std::pair<double, int>(dists.at(0), it->first));

        /*
        for(size_t histIdx = 0; histIdx < hists.size(); histIdx++) {
            double dist = compareHist(hists.at(histIdx), query, COMP_ALG);
            if((dist < minDist) && (dist < _threshold)) {
                minDist = dist;
                minClass = it->first;
            }

            if(dist < bestpreds[it->first]) {
                bestpreds[it->first] = dist;
            }
        }
        */
    }
    std::sort(bestpreds.begin(), bestpreds.end());
    minDist = bestpreds.at(0).first;
    minClass = bestpreds.at(0).second;
    
    /*
    std::cout << "\nBest Prediction by PID:\n";
    for(std::map<int, double>::const_iterator it = bestpreds.begin(); it != bestpreds.end(); ++it) {
        printf("[%d, %f]\n", it->first, it->second);
    }
    */
}



void xLBPH::predict(InputArray _src, int &minClass, double &minDist) const {
    
    CV_Assert((int)_labelinfo.size() > 0);
    CV_Assert((int)_histograms.size() > 0);

    /*
    if((int)_labelinfo.size() <= 0) {
        CV_Error(Error::StsError, "Given model path at '" + getModelPath() +"' already exists and doesn't look like an xLBPH model directory; refusing to overwrite for data safety.");
    }
    */

    Mat src = _src.getMat();
    // get the spatial histogram from input image
    Mat lbp_image = elbp(src, _radius, _neighbors);
    Mat query = spatial_histogram(
            lbp_image, /* lbp_image */
            static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), /* number of possible patterns */
            _grid_x, /* grid size x */
            _grid_y, /* grid size y */
            true /* normed histograms */);
    
    switch(_algToUse) {
        case 1: predict_avg(query, minClass, minDist); break;
        case 2: predict_avg_clustering(query, minClass, minDist); break;
        default: predict_std(query, minClass, minDist); break;
    }
    
    //printf("!!! Final Prediction: [%d, %f]\n", minClass, minDist);
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
