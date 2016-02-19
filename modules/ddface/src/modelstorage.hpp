
#ifndef __MODELSTORAGE_HPP 
#define __MODELSTORAGE_HPP

#include "precomp.hpp"

#include "cluster.hpp"

namespace cv { namespace face {

typedef struct AlgSettings {
   int radius;
   int neighbors;
   int grid_x;
   int grid_y;
} AlgSettings;


class ModelStorage {
private: 

   String _modelpath;
   String _modelname;
   AlgSettings _alg;

   // Basic Collection of Generic File/Directory Functions
   bool isDirectory(const String &filepath) const;
   bool isRegularFile(const String &filepath) const;
   bool fileExists(const String &filepath) const;
   String getFileName(const String &filepath) const;
   String getFileParent(const String &filepath) const;
   std::vector<String> listdir(const String &dirpath) const;
   bool mkdirs(const String &dirpath) const;
   bool rmr(const String &filepath) const;

   // Model Creation/Manipulation
   void setModelPath(String path);
   bool checkModel(const String &name, const String &path) const;

   // Utility Functions for File Paths
   String intToString(int num) const;
   String getLabelFilePrefix(int label) const;

public:

   ModelStorage(String path, int radius, int neighbors, int grid_x, int grid_y) {
      setModelPath(path);
      setAlgSettings(radius, neighbors, grid_x, grid_y);
   };

   ModelStorage(String path, AlgSettings alg) {
      setModelPath(path);
      setAlgSettings(alg);
   };

   
   void testCreation() const;
   void test() const;

   /* --==##: New Model Structure Definition :##==--
    *
    * /<modelname>
    *   - <modelname>-metadata.yml
    *   + <modelname>-labels
    *      - <modelname>-label-averages.bin
    *      +... <modelname>-label-<label>
    *            - <modelname>-label-<label>-histograms.bin
    *            - <modelname>-label-<label>-cluster-averages.bin
    *            - <modelname>-label-<label>-clusters.yml
    *
    */

   // Model Creation/Manipulation
   void setAlgSettings(int radius, int neighbors, int grid_x, int grid_y);
   void setAlgSettings(AlgSettings alg);

   // Model Writing
   bool create(bool overwrite = false) const;
   bool writeMetadata(AlgSettings alg, std::vector<int> &labels, std::vector<int> &numhists) const;
   bool writeMetadata(AlgSettings alg, std::map<int, int> &labelinfo) const;

   // Model Reading 
   //AlgSettings loadAlgSettings();
   //bool loadLabelInfo(std::map<int,int> &labelinfo) const; 
   void loadMetadata(AlgSettings &alg, std::map<int,int> &labelinfo);

   // Model Information
   bool isValidModel() const;
   bool exists() const;
   String getPath() const;
   String getName() const;
   AlgSettings getAlgSettings() const;
   int getHistogramSize() const;

   // Model File Getters
   String getMetadataFile() const;
   String getLabelsDir() const;
   String getLabelAveragesFile() const;
   String getLabelDir(int label) const;
   String getLabelHistogramsFile(int label) const;
   String getLabelClusterAveragesFile(int label) const;
   String getLabelClustersFile(int label) const;

   // Reading/Writing Histograms
   bool loadLabelHistograms(int label, std::vector<Mat> &histograms) const;
   bool saveLabelHistograms(int label, const std::vector<Mat> &histograms) const;
   bool updateLabelHistograms(int label, const std::vector<Mat> &histograms) const;

   bool loadLabelAverages(std::vector<Mat> &histograms) const;
   bool saveLabelAverages(const std::vector<Mat> &histograms) const;
   //bool updateLabelAverages(const std::vector<Mat> &histograms) const;

   bool readHistograms(const String &filename, std::vector<Mat> &histograms) const;
   bool writeHistograms(const String &filename, const std::vector<Mat> &histograms, bool appendhists) const;

   // Memory Mapping Histograms
   void mmapLabelHistograms(const std::map<int,int> &labelinfo, std::map<int, std::vector<Mat>> &histograms) const;
   void mmapLabelAverages(const std::map<int,int> &labelinfo, std::map<int, Mat> &histavgs) const;



   // Reading/Writing Clusters
   //bool loadLabelClusters(int label, std::vector<cluster::cluster_t> &clusters) const;
   //bool saveLabelClusters(int label, const std::vector<cluster::cluster_t> &clusters) const;
   bool saveClusters(const std::map<int, std::vector<cluster::cluster_t>> &clusters) const;
   bool writeLabelClusters(int label, const std::map<int, std::vector<cluster::cluster_t>>)
   // Memory Mapping Clusters
   void mmapLabelClusters(int label, std::vector<cluster::cluster_t> &clusters) const;
   void mmapClusters(const std::map<int,int> &labelinfo, std::map<int, std::vector<cluster::cluster_t>> &clusters) const;

};

}}

#endif

