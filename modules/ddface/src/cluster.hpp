
#ifndef __CLUSTER_HPP
#define __CLUSTER_HPP

#include "precomp.hpp"

namespace cv { namespace cluster {

    typedef std::pair<Mat, std::vector<Mat>> cluster_t;
    typedef std::set<int> idx_cluster_t;

    struct cluster_vars {
        double cluster_tierStep;
        int cluster_numTiers;
        int cluster_max_iterations;

        int mcl_iterations;
        int mcl_expansion_power;
        double mcl_inflation_power;
        double mcl_prune_min;
    };

    void clusterHistograms(const std::vector<Mat> &hists, std::vector<cluster_t> &clusters, const cluster_vars &vars);
    
}}

#endif

