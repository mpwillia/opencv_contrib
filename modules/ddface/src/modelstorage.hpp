
#ifndef __MODELSTORAGE_HPP 
#define __MODELSTORAGE_HPP

#include "precomp.hpp"

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

   // Basic Collection of Generic File/Directory Functions
   bool isDirectory(const String &filepath) const;
   bool isRegularFile(const String &filepath) const;
   bool fileExists(const String &filepath) const;
   String getFileName(const String &filepath) const;
   String getFileParent(const String &filepath) const;
   std::vector<String> listdir(const String &dirpath) const;
   bool mkdirs(const String &dirpath) const;
   bool rmr(const String &dirpath) const;

   // Model Creation/Manipulation
   void setModelPath(String path);
   bool checkModel(const String &name, const String &path) const;
   
   String intToString(int num) const;
   String getLabelFilePrefix(int label) const;

public:

   ModelStorage(String path) {
      setModelPath(path);
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
   bool create(bool overwrite = false) const;
   bool writeMetadata(AlgSettings alg, std::vector<int> &labels, std::vector<int> &numhists) const;
   bool writeMetadata(AlgSettings alg, std::map<int, int> &labelinfo) const;

   // Model Reading/Parsing
   AlgSettings getAlgSettings() const;
   bool getLabelInfo(std::map<int,int> &labelinfo) const;
   bool loadMetadata(AlgSettings &alg, std::map<int,int> &labelinfo) const;

   // Model Information
   bool isValidModel() const;
   bool exists() const;
   String getPath() const;
   String getName() const;

   // Model File Getters
   String getMetadataFile() const;
   String getLabelsDir() const;
   String getLabelAveragesFile() const;
   String getLabelDir(int label) const;
   String getLabelHistogramsFile(int label) const;
   String getLabelClusterAveragesFile(int label) const;
   String getLabelClustersFile(int label) const;

   // Reading/Writing Histograms
   /*
   bool loadHistograms(int label, std::vector<Mat> &histograms, int histSize) const;
   bool saveHistograms(int label, const std::vector<Mat> &histograms, int histSize) const;
   bool updateHistograms(int label, const std::vector<Mat> &histograms, int histSize) const;
   bool readHistograms(const String &filename, std::vector<Mat> &histograms, int histSize) const;
   bool writeHistograms(const String &filename, const std::vector<Mat> &histograms, bool appendhists, int histSize) const;
   */



};

}}

#endif

