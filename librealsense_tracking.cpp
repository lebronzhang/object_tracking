#include <pcl/tracking/tracking.h>
#include <pcl/tracking/particle_filter.h>
#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
#include <pcl/tracking/particle_filter_omp.h>

#include <pcl/tracking/coherence.h>
#include <pcl/tracking/distance_coherence.h>
#include <pcl/tracking/hsv_color_coherence.h>
#include <pcl/tracking/normal_coherence.h>

#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/io/librealsense_grabber.h>

#include <pcl/console/parse.h>
#include <pcl/common/time.h>
#include <pcl/common/centroid.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/io/pcd_io.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/surface/convex_hull.h>

#include <pcl/search/pcl_search.h>
#include <pcl/common/transforms.h>

#include <boost/format.hpp>

#include "define.h"
#include "Sampling.h"
#include "Ppfe.h"
#include "Icp.h"

#define FPS_CALC_BEGIN                          \
    static double duration = 0;                 \
    double start_time = pcl::getTime ();        \

#define FPS_CALC_END(_WHAT_)                    \
  {                                             \
    double end_time = pcl::getTime ();          \
    static unsigned count = 0;                  \
    if (++count == 10)                          \
    {                                           \
      std::cout << "Average framerate("<< _WHAT_ << "): " << double(count)/double(duration) << " Hz" <<  std::endl; \
      count = 0;                                                        \
      duration = 0.0;                                                   \
    }                                           \
    else                                        \
    {                                           \
      duration += end_time - start_time;        \
    }                                           \
  }

using namespace pcl::tracking;

template <typename PointType>
class LibRealSenseTracking
{
  public:
    typedef pcl::PointXYZRGBA RefPointType;
    typedef ParticleXYZRPY ParticleT;
    typedef pcl::PointCloud<PointType> Cloud;
    typedef pcl::PointCloud<RefPointType> RefCloud;
    typedef ParticleFilterTracker<RefPointType, ParticleT> ParticleFilter;
    typedef typename RefCloud::Ptr RefCloudPtr;
    typedef typename RefCloud::ConstPtr RefCloudConstPtr;
    typedef typename Cloud::Ptr CloudPtr;
    typedef typename Cloud::ConstPtr CloudConstPtr;
    typedef typename ParticleFilter::CoherencePtr CoherencePtr;
    typedef typename pcl::search::KdTree<PointType> KdTree;
    typedef typename KdTree::Ptr KdTreePtr;

    LibRealSenseTracking ()
    : viewer_ ("PCL LibRealSense Tracking With Recognition Viewer")
    , ne_ (thread_nr_)
    , new_cloud_ (false)
    , counter_ (0)
    {
      // Register KeyboardCallback
      viewer_.registerKeyboardCallback (&LibRealSenseTracking::keyboardCallback, *this);

      //Load model clouds
      if (use_fixed_model_)
      {
        reference_model_.reset (new Cloud);
        if (pcl::io::loadPCDFile (model_filename_, *reference_model_) < 0)
        {
          std::cout << "Error loading model cloud." << std::endl;
        }
        original_model_ = reference_model_;

        // Add the model to the filter so that the scene can be filtered using the model mean color
        if (to_filter_)
        {
          filter_ = new ColorSampling(filter_intensity_);
          filter_->AddCloud (*original_model_);
        }
      }
    }

    void
    initTracker ()
    {
      //Initialize tracker
      KdTreePtr tree (new KdTree (false));
      ne_.setSearchMethod (tree);//need this to set neighbors
      ne_.setRadiusSearch (0.03);//neighbor area search range
    
      std::vector<double> default_step_covariance = std::vector<double> (6, 0.015 * 0.015);
      default_step_covariance[3] *= 40.0;
      default_step_covariance[4] *= 40.0;
      default_step_covariance[5] *= 40.0;

      std::vector<double> initial_noise_covariance = std::vector<double> (6, 0.00001);
      std::vector<double> default_initial_mean = std::vector<double> (6, 0.0);

      if (use_fixed_)
      {
        //Initialize the scheduler and set the number of threads to use. 
        boost::shared_ptr<ParticleFilterOMPTracker<RefPointType, ParticleT> > tracker
          (new ParticleFilterOMPTracker<RefPointType, ParticleT> (thread_nr_));
        tracker_ = tracker;
      }
      else
      {
        boost::shared_ptr<KLDAdaptiveParticleFilterOMPTracker<RefPointType, ParticleT> > tracker
        (new KLDAdaptiveParticleFilterOMPTracker<RefPointType, ParticleT> (thread_nr_));
        tracker->setMaximumParticleNum (500);
        tracker->setDelta (0.99);
        tracker->setEpsilon (0.2);
        ParticleT bin_size;
        bin_size.x = 0.1f;
        bin_size.y = 0.1f;
        bin_size.z = 0.1f;
        bin_size.roll = 0.1f;
        bin_size.pitch = 0.1f;
        bin_size.yaw = 0.1f;
        tracker->setBinSize (bin_size);
        tracker_ = tracker;
      }
      tracker_->setTrans (Eigen::Affine3f::Identity ());
      tracker_->setStepNoiseCovariance (default_step_covariance);
      tracker_->setInitialNoiseCovariance (initial_noise_covariance);
      tracker_->setInitialNoiseMean (default_initial_mean);
      tracker_->setIterationNum (1);
    
      tracker_->setParticleNum (400);
      tracker_->setResampleLikelihoodThr(0.00);
      tracker_->setUseNormal (false);

      // setup coherences for tracker
      ApproxNearestPairPointCloudCoherence<RefPointType>::Ptr coherence = ApproxNearestPairPointCloudCoherence<RefPointType>::Ptr
        (new ApproxNearestPairPointCloudCoherence<RefPointType> ());
      // NearestPairPointCloudCoherence<RefPointType>::Ptr coherence = NearestPairPointCloudCoherence<RefPointType>::Ptr
      //   (new NearestPairPointCloudCoherence<RefPointType> ());
    
      boost::shared_ptr<DistanceCoherence<RefPointType> > distance_coherence
        = boost::shared_ptr<DistanceCoherence<RefPointType> > (new DistanceCoherence<RefPointType> ());
      coherence->addPointCoherence (distance_coherence);
    
      boost::shared_ptr<HSVColorCoherence<RefPointType> > color_coherence
        = boost::shared_ptr<HSVColorCoherence<RefPointType> > (new HSVColorCoherence<RefPointType> ());
      color_coherence->setWeight (0.1);
      coherence->addPointCoherence (color_coherence);
    
      //boost::shared_ptr<pcl::search::KdTree<RefPointType> > search (new pcl::search::KdTree<RefPointType> (false));
      boost::shared_ptr<pcl::search::Octree<RefPointType> > search (new pcl::search::Octree<RefPointType> (0.01));
      //boost::shared_ptr<pcl::search::OrganizedNeighbor<RefPointType> > search (new pcl::search::OrganizedNeighbor<RefPointType>);
      coherence->setSearchMethod (search);
      coherence->setMaximumDistance (0.01);
      tracker_->setCloudCoherence (coherence);
    }

    void
    keyboardCallback (const pcl::visualization::KeyboardEvent& event, void*)
    {
      if (event.keyDown ())
      {
        if (event.getKeyCode () == 's' || event.getKeyCode () == 'S')
        {
          pcl::io::savePCDFileASCII ("model.pcd", *reference_model_);
          std::cerr << "Saved " << reference_model_->points.size () << " data points to model.pcd." << std::endl;
        }

        if (event.getKeyCode () == 't' || event.getKeyCode () == 'T')
        {
          std::cerr << "keyboardCallback counter_: " << counter_ << std::endl;
          counter_ = 9;
          if (use_fixed_model_)
            std::cerr << "Switch to Recognition mode" << std::endl;
          else
            std::cerr << "Switch to Segmentation mode" << std::endl;
        }
      }
    }

    bool
    drawParticles (pcl::visualization::PCLVisualizer& viz)
    {
      ParticleFilter::PointCloudStatePtr particles = tracker_->getParticles ();
      if (particles)
      {
        if (visualize_particles_)
        {
          pcl::PointCloud<pcl::PointXYZ>::Ptr particle_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
          for (size_t i = 0; i < particles->points.size (); i++)
          {
            pcl::PointXYZ point;
          
            point.x = particles->points[i].x;
            point.y = particles->points[i].y;
            point.z = particles->points[i].z;
            particle_cloud->points.push_back (point);
          }

          {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue_color (particle_cloud, 250, 99, 71);
            if (!viz.updatePointCloud (particle_cloud, blue_color, "particle cloud"))
            {
              viz.addPointCloud (particle_cloud, blue_color, "particle cloud");
              viz.resetCameraViewpoint ("particle cloud");
            }
          }
        }
        return true;
      }
      else
      {
        PCL_WARN ("no particles\n");
        return false;
      }
    }

    void
    drawResult (pcl::visualization::PCLVisualizer& viz)
    {
      ParticleXYZRPY result = tracker_->getResult ();
      Eigen::Affine3f transformation = tracker_->toEigenMatrix (result);
      // move a little bit for better visualization
      transformation.translation () += Eigen::Vector3f (0.0f, 0.0f, -0.005f);
      RefCloudPtr result_cloud (new RefCloud ());

      if (!visualize_non_downsample_)
        pcl::transformPointCloud<RefPointType> (*(tracker_->getReferenceCloud ()), *result_cloud, transformation);
      else
        pcl::transformPointCloud<RefPointType> (*reference_, *result_cloud, transformation); // to see if it is useless

      {
        pcl::visualization::PointCloudColorHandlerCustom<RefPointType> red_color (result_cloud, 0, 0, 255);
        if (!viz.updatePointCloud (result_cloud, red_color, "resultcloud"))
        {
          viz.addPointCloud (result_cloud, red_color, "resultcloud");
          viz.resetCameraViewpoint ("resultcloud");
        }
      }

    }

    void
    viz_cb (pcl::visualization::PCLVisualizer& viz)
    {
      boost::mutex::scoped_lock lock (mtx_);

      if (!cloud_pass_)
      {
        boost::this_thread::sleep (boost::posix_time::seconds (1));
        return;
      }

      if (new_cloud_  && cloud_pass_downsampled_)
      {
        CloudPtr cloud_pass;
        if (!visualize_non_downsample_)
          cloud_pass = cloud_pass_downsampled_;
        else
          cloud_pass = cloud_pass_;
      
        if (!viz.updatePointCloud (cloud_pass, "cloudpass"))
        {
          viz.addPointCloud (cloud_pass, "cloudpass");
          viz.resetCameraViewpoint ("cloudpass");
        }

        bool ret = drawParticles (viz);
        if (ret)
        {
          drawResult (viz);
          viz.removeShape ("Tracking");
          viz.addText ((boost::format ("Tracking:     %f fps") % (1.0 / computation_time_)).str (),
                     10, 20, 20, 1.0, 1.0, 1.0, "Tracking");
        }
      }
      new_cloud_ = false;
    }

    void
    filterPassThrough (const CloudConstPtr &cloud, Cloud &result)
    {
      FPS_CALC_BEGIN;
      pcl::PassThrough<PointType> pass;
      //CloudPtr res (new pcl::PointCloud<pcl::PointXYZRGBA>);
      pass.setFilterFieldName ("z");
      pass.setFilterLimits (-1.0, 0.0);//changed
      pass.setKeepOrganized (false);
      pass.setInputCloud (cloud);
      pass.filter (result);
      FPS_CALC_END("filterPassThrough");
    }

    void
    euclideanSegment (const CloudConstPtr &cloud,
                           std::vector<pcl::PointIndices> &cluster_indices)
    {
      FPS_CALC_BEGIN;
      pcl::EuclideanClusterExtraction<PointType> ec;
      KdTreePtr tree (new KdTree ());
    
      ec.setClusterTolerance (0.05); // 2cm
      ec.setMinClusterSize (50);
      ec.setMaxClusterSize (25000);
      //ec.setMaxClusterSize (400);
      ec.setSearchMethod (tree);
      ec.setInputCloud (cloud);
      ec.extract (cluster_indices);
      FPS_CALC_END("euclideanSegmentation");
    }
  
    void
    gridSample (const CloudConstPtr &cloud, Cloud &result, double leaf_size = 0.01)
    {
      FPS_CALC_BEGIN;
      double start = pcl::getTime ();
      pcl::VoxelGrid<PointType> grid;
      //pcl::ApproximateVoxelGrid<PointType> grid;
      grid.setLeafSize (float (leaf_size), float (leaf_size), float (leaf_size));
      grid.setInputCloud (cloud);
      grid.filter (result);
      //result = *cloud;
      double end = pcl::getTime ();
      downsampling_time_ = end - start;
      FPS_CALC_END("gridSample");
    }
  
    void
    gridSampleApprox (const CloudConstPtr &cloud, Cloud &result, double leaf_size = 0.01)
    {
      FPS_CALC_BEGIN;
      double start = pcl::getTime ();
      //pcl::VoxelGrid<PointType> grid;
      pcl::ApproximateVoxelGrid<PointType> grid;
      grid.setLeafSize (static_cast<float> (leaf_size), static_cast<float> (leaf_size), static_cast<float> (leaf_size));
      grid.setInputCloud (cloud);
      grid.filter (result);
      //result = *cloud;
      double end = pcl::getTime ();
      downsampling_time_ = end - start;
      FPS_CALC_END("gridSample");
    }
  
    void
    planeSegmentation (const CloudConstPtr &cloud,
                            pcl::ModelCoefficients &coefficients,
                            pcl::PointIndices &inliers)
    {
      FPS_CALC_BEGIN;
      pcl::SACSegmentation<PointType> seg;
      seg.setOptimizeCoefficients (true);
      seg.setModelType (pcl::SACMODEL_PLANE);
      seg.setMethodType (pcl::SAC_RANSAC);
      seg.setMaxIterations (1000);
      seg.setDistanceThreshold (0.03);
      seg.setInputCloud (cloud);
      seg.segment (inliers, coefficients);
      FPS_CALC_END("planeSegmentation");
    }

    void
    planeProjection (const CloudConstPtr &cloud,
                          Cloud &result,
                          const pcl::ModelCoefficients::ConstPtr &coefficients)
    {
      FPS_CALC_BEGIN;
      pcl::ProjectInliers<PointType> proj;
      proj.setModelType (pcl::SACMODEL_PLANE);
      proj.setInputCloud (cloud);
      proj.setModelCoefficients (coefficients);
      proj.filter (result);
      FPS_CALC_END("planeProjection");
    }

    void
    convexHull (const CloudConstPtr &cloud,
                     Cloud &,
                     std::vector<pcl::Vertices> &hull_vertices)
    {
      FPS_CALC_BEGIN;
      pcl::ConvexHull<PointType> chull;
      chull.setInputCloud (cloud);
      chull.reconstruct (*cloud_hull_, hull_vertices);
      FPS_CALC_END("convexHull");
    }

    void
    normalEstimation (const CloudConstPtr &cloud,
                           pcl::PointCloud<pcl::Normal> &result)
    {
      FPS_CALC_BEGIN;
      ne_.setInputCloud (cloud);
      ne_.compute (result);
      FPS_CALC_END("normalEstimation");
    }
  
    void
    tracking (const RefCloudConstPtr &cloud)
    {
      double start = pcl::getTime ();
      FPS_CALC_BEGIN;
      tracker_->setInputCloud (cloud);
      tracker_->compute ();
      double end = pcl::getTime ();
      FPS_CALC_END("tracking");
      tracking_time_ = end - start;
    }

    void
    addNormalToCloud (const CloudConstPtr &cloud,
                           const pcl::PointCloud<pcl::Normal>::ConstPtr &,
                           RefCloud &result)
    {
      result.width = cloud->width;
      result.height = cloud->height;
      result.is_dense = cloud->is_dense;
      for (size_t i = 0; i < cloud->points.size (); i++)
      {
        RefPointType point;
        point.x = cloud->points[i].x;
        point.y = cloud->points[i].y;
        point.z = cloud->points[i].z;
        point.rgba = cloud->points[i].rgba;
        // point.normal[0] = normals->points[i].normal[0];
        // point.normal[1] = normals->points[i].normal[1];
        // point.normal[2] = normals->points[i].normal[2];
        result.points.push_back (point);
      }
    }

    void
    extractNonPlanePoints (const CloudConstPtr &cloud,
                                const CloudConstPtr &cloud_hull,
                                Cloud &result)
    {
      pcl::ExtractPolygonalPrismData<PointType> polygon_extract;
      pcl::PointIndices::Ptr inliers_polygon (new pcl::PointIndices ());
      polygon_extract.setHeightLimits (0.01, 10.0);
      polygon_extract.setInputPlanarHull (cloud_hull);
      polygon_extract.setInputCloud (cloud);
      polygon_extract.segment (*inliers_polygon);
      {
        pcl::ExtractIndices<PointType> extract_positive;
        extract_positive.setNegative (false);
        extract_positive.setInputCloud (cloud);
        extract_positive.setIndices (inliers_polygon);
        extract_positive.filter (result);
      }
    }

    void
    removeZeroPoints (const CloudConstPtr &cloud,
                           Cloud &result)
    {
      for (size_t i = 0; i < cloud->points.size (); i++)
      {
        PointType point = cloud->points[i];
        if (!(fabs(point.x) < 0.01 &&
              fabs(point.y) < 0.01 &&
              fabs(point.z) < 0.01) &&
            !pcl_isnan(point.x) &&
            !pcl_isnan(point.y) &&
            !pcl_isnan(point.z))
          result.points.push_back(point);
      }

      result.width = static_cast<pcl::uint32_t> (result.points.size ());
      result.height = 1;
      result.is_dense = true;
    }
  
    void
    extractSegmentCluster (const CloudConstPtr &cloud,
                                const std::vector<pcl::PointIndices> cluster_indices,
                                const int segment_index,
                                Cloud &result)
    {
      pcl::PointIndices segmented_indices = cluster_indices[segment_index];
      for (size_t i = 0; i < segmented_indices.indices.size (); i++)
      {
        PointType point = cloud->points[segmented_indices.indices[i]];
        result.points.push_back (point);
      }
      result.width = pcl::uint32_t (result.points.size ());
      result.height = 1;
      result.is_dense = true;
    }

    void
    cloud_cb (const CloudConstPtr &cloud)
    {
      int i=0;
      boost::mutex::scoped_lock lock (mtx_);
      double start = pcl::getTime ();
      FPS_CALC_BEGIN;
      cloud_pass_.reset (new Cloud);
      cloud_pass_downsampled_.reset (new Cloud);
      pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
      pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
      filterPassThrough (cloud, *cloud_pass_);
      if (counter_ < 10)
      {
        gridSample (cloud_pass_, *cloud_pass_downsampled_, downsampling_grid_size_);
        counter_++;
        return;
      }
      else if (counter_ == 10)
      {
        initTracker ();
        if (!use_fixed_model_)  // If there is no available model file, segment first, counter = 10, when press 'r' or 'R', set counter = 10, segmentation again
        {
          cloud_pass_downsampled_ = cloud_pass_;
          CloudPtr target_cloud;
          if (use_convex_hull_)
          {
            planeSegmentation (cloud_pass_downsampled_, *coefficients, *inliers);
            if (inliers->indices.size () > 3)
            {
              CloudPtr cloud_projected (new Cloud);
              cloud_hull_.reset (new Cloud);
              nonplane_cloud_.reset (new Cloud);

              planeProjection (cloud_pass_downsampled_, *cloud_projected, coefficients);
              convexHull (cloud_projected, *cloud_hull_, hull_vertices_);
              extractNonPlanePoints (cloud_pass_downsampled_, cloud_hull_, *nonplane_cloud_);
              target_cloud = nonplane_cloud_;
            }
            else
            {
              PCL_WARN ("cannot segment plane\n");
            }
          }
          else
          {
            PCL_WARN ("without plane segmentation\n");
            target_cloud = cloud_pass_downsampled_;
          }

          if (target_cloud != NULL)
          {
            PCL_INFO ("segmentation, please wait...\n");
            std::vector<pcl::PointIndices> cluster_indices;
            euclideanSegment (target_cloud, cluster_indices);
            if (cluster_indices.size () > 0)
            {
              // select the cluster to track
              CloudPtr temp_cloud (new Cloud);
              extractSegmentCluster (target_cloud, cluster_indices, 0, *temp_cloud);
              Eigen::Vector4f c;
              pcl::compute3DCentroid<RefPointType> (*temp_cloud, c);
              int segment_index = 0;
              double segment_distance = c[0] * c[0] + c[1] * c[1];
              for (size_t i = 1; i < cluster_indices.size (); i++)
              {
                temp_cloud.reset (new Cloud);
                extractSegmentCluster (target_cloud, cluster_indices, int (i), *temp_cloud);
                pcl::compute3DCentroid<RefPointType> (*temp_cloud, c);
                double distance = c[0] * c[0] + c[1] * c[1];
                if (distance < segment_distance)
                {
                  segment_index = int (i);
                  segment_distance = distance;
                }
              }
              // Set up model
              segmented_cloud_.reset (new Cloud);
              extractSegmentCluster (target_cloud, cluster_indices, segment_index, *segmented_cloud_);
              reference_model_ = segmented_cloud_;
            }
            else
            {
              PCL_WARN ("euclidean segmentation failed\n");
            }
          }
        }
        else  // Else there is an available model file, recognition, counter = 10, when press 'r' or 'R', set counter = 10, recognition again
        {
          std::cerr << "Recognition begin" << std::endl;
          // Delete the main plane to reduce the number of points in the scene point cloud
          if (segment_)
            cloud_pass_ = FindAndSubtractPlane (cloud_pass_, segmentation_threshold_, segmentation_iterations_);

          // Filter the scene using the mean model color calculated before
          if (to_filter_)
          {
            filter_->FilterPointCloud (*cloud_pass_, *cloud_pass_);
          }

          if (ppfe_)
          {
            // Calculate the model keypoints using the specified method
            ppfe_estimator_ = new Ppfe (original_model_);
            model_keypoints_ = ppfe_estimator_->GetModelKeypoints ();
            cluster_ = ppfe_estimator_->GetCluster (cloud_pass_);
            scene_keypoints_ = ppfe_estimator_->GetSceneKeypoints ();
            std::cout << "\tFound " << std::get < 0 > (cluster_).size () << " model instance/instances " << std::endl;
            if(std::get < 0 > (cluster_).size () > 0 && cloud_pass_->size () != 0)
            {
              if(use_icp_)
              {
                CloudPtr rotated_model (new Cloud ());
                std::cout << "\t USING ICP"<<std::endl;
                pcl::transformPointCloud (*original_model_, *rotated_model, (std::get < 0 > (cluster_)[0]));
                Eigen::Matrix4f tmp = Eigen::Matrix4f::Identity();
                if (cloud_pass_->size ()*5 > rotated_model->size ())
                  for(int i = 0; i< icp_iteration_; ++i)
                  {
                    icp_.Align (rotated_model, cloud_pass_);
                    tmp = icp_.transformation_ * tmp;
                  }
                pcl::transformPointCloud (*original_model_, *reference_model_, (std::get < 0 > (cluster_)[0]));
              }
            }
          }
        }
        RefCloudPtr nonzero_ref (new RefCloud);
        removeZeroPoints (reference_model_, *nonzero_ref);
        Eigen::Vector4f c;

        PCL_INFO ("calculating cog\n");
        RefCloudPtr transed_ref (new RefCloud);
        pcl::compute3DCentroid<RefPointType> (*nonzero_ref, c);
        Eigen::Affine3f trans = Eigen::Affine3f::Identity ();
        trans.translation ().matrix () = Eigen::Vector3f (c[0], c[1], c[2]);
        //pcl::transformPointCloudWithNormals<RefPointType> (*reference_model_, *transed_ref, trans.inverse());
        pcl::transformPointCloud<RefPointType> (*nonzero_ref, *transed_ref, trans.inverse());
        CloudPtr transed_ref_downsampled (new Cloud);
        gridSample (transed_ref, *transed_ref_downsampled, downsampling_grid_size_);
        tracker_->setReferenceCloud (transed_ref_downsampled);
        tracker_->setTrans (trans);
        reference_ = transed_ref;
        tracker_->setMinIndices (int (reference_model_->points.size ()) / 2);

      }
      else  // Track, counter >= 11
      {
        gridSampleApprox (cloud_pass_, *cloud_pass_downsampled_, downsampling_grid_size_);
        tracking (cloud_pass_downsampled_);
      }
      new_cloud_ = true;
      double end = pcl::getTime ();
      computation_time_ = end - start;
      FPS_CALC_END("computation");
      counter_++;
      if (counter_ > 1000000) counter_ = 11;
    }

    void
    run ()
    {
      pcl::Grabber* interface = new pcl::LibRealSenseGrabber (device_id_);
      boost::function<void (const CloudConstPtr&)> f =
      boost::bind (&LibRealSenseTracking::cloud_cb, this, _1);
      interface->registerCallback (f);
    
      viewer_.runOnVisualizationThread (boost::bind(&LibRealSenseTracking::viz_cb, this, _1), "viz_cb");
    
      interface->start ();
      
      while (!viewer_.wasStopped ())
        boost::this_thread::sleep(boost::posix_time::seconds(1));
      interface->stop ();
    }

    pcl::visualization::CloudViewer viewer_;
    CloudPtr cloud_pass_;
    CloudPtr cloud_pass_downsampled_;
    CloudPtr reference_;//transformed Cloud
    CloudPtr cloud_hull_;
    CloudPtr nonplane_cloud_;
    CloudPtr segmented_cloud_;
    CloudPtr model_keypoints_;
    CloudPtr scene_keypoints_;
    RefCloudPtr reference_model_;
    RefCloudPtr original_model_;
    ColorSampling *filter_;
    Ppfe *ppfe_estimator_;
    ClusterType cluster_;
    ICP icp_;
    std::vector<pcl::Vertices> hull_vertices_;
    boost::mutex mtx_;
    bool new_cloud_;
    pcl::NormalEstimationOMP<PointType, pcl::Normal> ne_; // to store threadpool
    boost::shared_ptr<ParticleFilter> tracker_;
    int counter_;
    double tracking_time_;
    double computation_time_;
    double downsampling_time_;

};

void
usage (char** argv)
{
  std::cout << std::endl;
  std::cout << "****************************************************************************" << std::endl;
  std::cout << "*                                                                          *" << std::endl;
  std::cout << "*              LIBREALSENSE OBJECT TRACKING - Usage Guide                  *" << std::endl;
  std::cout << "*                                                                          *" << std::endl;
  std::cout << "****************************************************************************" << std::endl;
  std::cout << std::endl;
  std::cout << " Usage: " << argv[0] << " <device_id> [options]\n\n";
  std::cout << " Default: Use the nearest object of the camera as the target tracking object." << std::endl;
  std::cout << " --model <model_file.pcd>: use the segmented model to track." << std::endl;
  std::cout << " -P: not visualizing particle cloud." << std::endl;
  std::cout << " -h or --help: print Usage Guide." << std::endl;
  std::cout << std::endl;
  std::cout << " Press 't' or 'T' to switch to recognition mode." << std::endl;
  std::cout << " Press 's' or 'S' to save the target model pcd file into your current work directory, named as \"model.pcd\"." << std::endl;
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  device_id_ = std::string (argv[1]);

  if (pcl::console::find_argument (argc, argv, "-P") > 0)
  {
    visualize_particles_ = false;
  }

  if (pcl::console::find_argument (argc, argv, "-fixed") > 0)
  {
    use_fixed_ = true;
  }

  if (pcl::console::parse_argument (argc, argv, "--model", model_filename_) != -1)
  {
    use_fixed_model_ = true;
  }

  if (device_id_ == "--help" || device_id_ == "-h")
  {
    usage (argv);
    exit (1);
  }
  

  LibRealSenseTracking<pcl::PointXYZRGBA> v;
  v.run ();
  return 0;
}