
#ifndef __CLUSTER_HPP
#define __CLUSTER_HPP

#include "precomp.hpp"

namespace cv { namespace face {
   
   typedef cluster_t std::pair<Mat, std::vector<Mat>>;
   
   //void clusterHistograms(const std::vector<Mat> &hists, std::vector<cluster_t>>);

}}

#endif

