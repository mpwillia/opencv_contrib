#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace xfeatures2d;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef perf::TestBaseWithParam<std::string> msd;

#define MSD_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png"

PERF_TEST_P(msd, detect, testing::Values(MSD_IMAGES))
{
    string filename = getDataPath(GetParam());
    Mat frame = imread(filename, IMREAD_GRAYSCALE);

    if (frame.empty())
        FAIL() << "Unable to load source image " << filename;

    Mat mask;
    declare.in(frame);
    Ptr<MSDDetector> detector = MSDDetector::create();
    vector<KeyPoint> points;

    TEST_CYCLE() detector->detect(frame, points, mask);

    sort(points.begin(), points.end(), comparators::KeypointGreater());
    SANITY_CHECK_KEYPOINTS(points, 1e-3);
}
