
#include "cluster.hpp"
#include "mcl.hpp"
#include "tbb/tbb.h"

#define COMP_ALG HISTCMP_CHISQR_ALT

namespace cv { namespace cluster {

    // Calculates the weights between each histogram and puts them in weights
    void calc_weights(Mat &dists, Mat &weights, double tierStep, int numTiers) {
        weights.create(dists.rows, dists.cols, dists.type());

        // calculate tiers and weights
        //for(size_t i = 0; i < dists.rows; i++) {
        tbb::parallel_for(0, dists.rows, 1, [&](int i) {
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
        });
    }

    // Finds clusters for the given label's dists and puts the MCL mat in mclmat
    void cluster_dists(Mat &dists, Mat &mclmat, double r, const cluster_vars &vars) {
        //printf("\t\t\t - clustering dists...\n");
        mclmat.create(dists.rows, dists.cols, dists.type());

        // find weights
        calc_weights(dists, mclmat, vars.cluster_tierStep, vars.cluster_numTiers);

        // iterate
        mcl::cluster(mclmat, vars.mcl_iterations, vars.mcl_expansion_power, r, vars.mcl_prune_min);
    }

    // Interprets a given MCL matrix as clusters
    void interpret_clusters(Mat &mclmat, std::vector<idx_cluster_t> &clusters) {

        //printf("\t\t\t - interpreting clusters...\n");
        // interpret clusters
        std::map<int, idx_cluster_t> clusters_map;
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
        
        for(std::map<int, idx_cluster_t>::const_iterator it = clusters_map.begin(); it != clusters_map.end(); it++) {
            if(!it->second.empty())
                clusters.push_back(it->second);
        }
    }

    void find_optimal_clustering(Mat &dists, std::vector<idx_cluster_t> &idxClusters, const cluster_vars &vars) {

        int optimalClusters = floor(sqrt(dists.rows));
        
        Mat initial;
        double r = vars.mcl_inflation_power;
        double r_dec = r - 1;
        double multStep = 0.75;

        cluster_dists(dists, initial, r, vars);
        interpret_clusters(initial, idxClusters);
        
        double clusterRatio = (int)idxClusters.size() / (double)optimalClusters;
        int iterations = vars.cluster_max_iterations;
            
        if(clusterRatio < 1) {
            // want more clusters - larger r
            Mat prevmat;
            for(int i = 0; i < iterations; i++) {
                r_dec *= (1 - multStep) + 1;

                Mat mclmat;
                cluster_dists(dists, mclmat, 1 + r_dec, vars);
                idxClusters.clear();
                interpret_clusters(mclmat, idxClusters);
                clusterRatio = (int)idxClusters.size() / (double)optimalClusters;

                if(clusterRatio > 1) {
                    interpret_clusters(prevmat, idxClusters);
                    break;
                } 
                else if(clusterRatio == 1.0) {
                    break; 
                } 
                else {
                    prevmat = mclmat; 
                }
            } 

        } 
        else if(clusterRatio > 1) {
            // want fewer clusters - smaller r 

            for(int i = 0; i < iterations; i++) {
                r_dec *= multStep;

                Mat mclmat;
                cluster_dists(dists, mclmat, 1 + r_dec, vars);
                idxClusters.clear();
                interpret_clusters(mclmat, idxClusters);
                clusterRatio = (int)idxClusters.size() / (double)optimalClusters;

                if(clusterRatio <= 1.0) {
                    break; 
                } 
            } 
        } 
        
        /*
        int optimalClustersMax = ceil(sqrt(dists.rows));
        int optimalClustersMin = floor(sqrt(dists.rows));
        int optimalCase = (int)ceil(sqrt((int)dists.rows)*2);
        double optimalRatio = optimalCase / (double)dists.rows;
        */

        /*
        printf("Optimal Case Checks: %d\n", optimalCase);
        printf("Optimal Check Ratio: %.3f\n", optimalRatio);
        printf("Optimal Clusters: %d - %d\n\n", optimalClustersMin, optimalClustersMax);
        */
        
        /*
        Mat initial;
        double r = vars.mcl_inflation_power;

        cluster_dists(dists, initial, r, vars);
        interpret_clusters(initial, idxClusters);
        
        int checkClusters = (int)idxClusters.size();
        int iterations = vars.cluster_max_iterations;
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
            cluster_dists(dists, mclmat, r, vars);
            idxClusters.clear();
            interpret_clusters(mclmat, idxClusters);
            checkClusters = (int)idxClusters.size();
        }
        */
    }

    void averageHistograms(const std::vector<Mat> &hists, Mat &histavg) {
        histavg = Mat::zeros(1, hists.at(0).cols, CV_64FC1);

        for(size_t idx = 0; idx < hists.size(); idx++) {
            Mat dst;
            hists.at((int)idx).convertTo(dst, CV_64FC1);
            histavg += dst; 
        }
        histavg /= (int)hists.size();
        histavg.convertTo(histavg, CV_32FC1);
    }

    void clusterHistograms(const std::vector<Mat> &hists, std::vector<cluster_t> &clusters, const cluster_vars &vars) {

        // calculate hist distances
        Mat dists = Mat::zeros((int)hists.size(), (int)hists.size(), CV_64FC1);
        tbb::parallel_for(0, (int)hists.size()-1, 1, 
            [&hists, &dists](int i) {
                tbb::parallel_for(i, (int)hists.size(), 1, 
                    [&hists, &dists, &i](int j) {
                        double dist = compareHist(hists.at(i), hists.at(j), COMP_ALG);
                        dists.at<double>(i, j) = dist;
                        dists.at<double>(j, i) = dist;
                    } 
                );
            } 
        );
        
        // find optimal clusters
        std::vector<idx_cluster_t> idxClusters;
        find_optimal_clustering(dists, idxClusters, vars);

        // convert from idx_cluster_t to cluster_t
        for(size_t i = 0; i < idxClusters.size(); i++) {
            idx_cluster_t cluster = idxClusters.at((int)i);
            
            std::vector<Mat> clusterHists;
            Mat clusterAvg;

            for(idx_cluster_t::const_iterator it = cluster.begin(); it != cluster.end(); it++) {
                clusterHists.push_back(hists.at(*it));
            }
            
            averageHistograms(clusterHists, clusterAvg);

            clusters.push_back(cluster_t(clusterAvg, clusterHists));
        }
    }

}}

