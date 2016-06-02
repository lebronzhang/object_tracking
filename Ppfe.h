#ifndef OBJECT_TWR_PPFE_H
#define OBJECT_TWR_PPFE_H

#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/registration/ppf_registration.h>
#include <pcl/filters/extract_indices.h>

#include "define.h"
//define.h
/*#include <pcl/features/board.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>*/

/*typedef pcl::PointXYZ XYZType;
typedef std::tuple<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >, std::vector<pcl::Correspondences>> ClusterType;
typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;*/

class Ppfe
{
  public:
    pcl::SACSegmentation<XYZType> seg_;
    pcl::ExtractIndices<XYZType> extract_;
    pcl::PointCloud<XYZType>::Ptr model_xyz_;
    pcl::ModelCoefficients::Ptr coefficients_;
    pcl::PointIndices::Ptr inliers_;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_model_input_;
    pcl::PointCloud<pcl::PPFSignature>::Ptr cloud_model_ppf_;
    pcl::PPFRegistration<pcl::PointNormal, pcl::PointNormal> ppf_registration_;
    pcl::PPFHashMapSearch::Ptr hashmap_search_;
    pcl::PointCloud<XYZType>::Ptr cloud_scene_;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_scene_input_;
    pcl::PointCloud<pcl::PointNormal> cloud_output_subsampled_;
    Eigen::Matrix4f mat_;
    ClusterType cluster_;
    unsigned nr_points_;
    /*float sac_seg_iter;
    float reg_sampling_rate;
    float reg_clustering_threshold;
    float sac_seg_distance;
    float max_inliers;
    const Eigen::Vector4f SUBSAMPLING_LEAF_SIZE;*/

    Ppfe (pcl::PointCloud<PointType>::Ptr model);

    ClusterType
    GetCluster (pcl::PointCloud<PointType>::Ptr scene);

    pcl::PointCloud<PointType>::Ptr
    GetModelKeypoints ();

    pcl::PointCloud<PointType>::Ptr
    GetSceneKeypoints ();
};

pcl::PointCloud<pcl::PointNormal>::Ptr
SubsampleAndCalculateNormals (pcl::PointCloud<XYZType>::Ptr cloud);

#endif