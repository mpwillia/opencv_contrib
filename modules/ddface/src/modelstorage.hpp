
#ifndef __MODELSTORAGE_HPP 
#define __MODELSTORAGE_HPP

#include "precomp.hpp"

namespace cv { namespace face {

class ModelStorage {
private: 

   String _modelpath;
   String _modelname;

   void setModelPath(String path);

public:

   ModelStorage(String path) {
      setModelPath(path);
   };

   void test() const;
   bool isValidModel() const;
   bool modelExists() const;
   String getModelPath() const;
   String getModelName() const;

};

}}

#endif

