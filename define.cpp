#include "define.h"

const Eigen::Vector4f SUBSAMPLING_LEAF_SIZE_ (0.01f, 0.01f, 0.01f, 0.00f);

bool use_convex_hull_ (true);
bool visualize_non_downsample_ (true);
bool visualize_particles_ (true);
bool use_fixed_model_ (false);
bool use_fixed_ (false);
std::string device_id_;
std::string model_filename_;
int thread_nr_ (8); //threads number
double downsampling_grid_size_ (0.01);

int icp_iteration_ (10);
float filter_intensity_ (0.04);
bool to_filter_ (true);
bool ppfe_ (true);
float sac_seg_iter_ (1000);
float reg_sampling_rate_ (10);
float reg_clustering_threshold_ (0.2);
float sac_seg_distance_ (0.05);
float normal_estimation_search_radius_ (0.05);
float max_inliers_ (40000);
bool use_generalized_icp_ (false);
bool use_icp_ (true);
bool segment_ (true);
float segmentation_threshold_ (0.01);
int segmentation_iterations_ (1000);