#ifndef OBJECT_TWR_DEFINE_H
#define OBJECT_TWR_DEFINE_H

#include <pcl/features/board.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
//defines for other objects
typedef pcl::PointXYZRGBA PointType;
typedef pcl::PointCloud<PointType>::ConstPtr CloudConstPtr;
typedef pcl::Normal NormalType;
typedef pcl::PointXYZ XYZType;
typedef pcl::PointXYZI XYZIType;
typedef pcl::ReferenceFrame RFType;
typedef std::tuple<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >, std::vector<pcl::Correspondences>> ClusterType;
typedef std::tuple<float, float> error;

extern const Eigen::Vector4f SUBSAMPLING_LEAF_SIZE_;

extern bool use_convex_hull_;
extern bool visualize_non_downsample_;
extern bool visualize_particles_;
extern bool use_fixed_model_;
extern bool use_fixed_;
extern std::string device_id_;
extern std::string model_filename_;
extern int thread_nr_;
extern double downsampling_grid_size_;

extern int icp_iteration_;
extern float filter_intensity_;
extern bool to_filter_;
extern bool ppfe_;
extern float sac_seg_iter_;
extern float reg_sampling_rate_;
extern float reg_clustering_threshold_;
extern float sac_seg_distance_;
extern float normal_estimation_search_radius_;
extern float max_inliers_;
extern bool use_generalized_icp_;
extern bool use_icp_;
extern bool segment_;
extern float segmentation_threshold_;
extern int segmentation_iterations_;

#endif