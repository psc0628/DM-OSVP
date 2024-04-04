#pragma once
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <direct.h>
#include <fstream>  
#include <stdio.h>  
#include <string>  
#include <sstream>  
#include <vector> 
#include <thread>
#include <chrono>
#include <atomic>
#include <ctime> 
#include <cmath>
#include <mutex>
#include <map>
#include <set>
#include <io.h>
#include <memory>
#include <functional>
#include <cassert>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/common/io.h>
#include <pcl/surface/gp3.h>
#include <pcl/geometry/polygon_mesh.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>
#include <vtkVersion.h>
#include <vtkOBJReader.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <vtkColorTransferFunction.h>
#include <vtkSmartPointer.h>
#include <vtkLookupTable.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

using namespace std;

//Camera Definition
/** \brief Distortion model: defines how pixel coordinates should be mapped to sensor coordinates. */
typedef enum rs2_distortion
{
	RS2_DISTORTION_NONE, /**< Rectilinear images. No distortion compensation required. */
	RS2_DISTORTION_MODIFIED_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except that tangential distortion is applied to radially distorted points */
	RS2_DISTORTION_INVERSE_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except undistorts image instead of distorting it */
	RS2_DISTORTION_FTHETA, /**< F-Theta fish-eye distortion model */
	RS2_DISTORTION_BROWN_CONRADY, /**< Unmodified Brown-Conrady distortion model */
	RS2_DISTORTION_KANNALA_BRANDT4, /**< Four parameter Kannala Brandt distortion model */
	RS2_DISTORTION_COUNT                   /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
} rs2_distortion;
/** \brief Video stream intrinsics. */
typedef struct rs2_intrinsics
{
	int           width;     /**< Width of the image in pixels */
	int           height;    /**< Height of the image in pixels */
	float         ppx;       /**< Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge */
	float         ppy;       /**< Vertical coordinate of the principal point of the image, as a pixel offset from the top edge */
	float         fx;        /**< Focal length of the image plane, as a multiple of pixel width */
	float         fy;        /**< Focal length of the image plane, as a multiple of pixel height */
	rs2_distortion model;    /**< Distortion model of the image */
	float         coeffs[5]; /**< Distortion coefficients */
} rs2_intrinsics;
/* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
static void rs2_project_point_to_pixel(float pixel[2], const struct rs2_intrinsics* intrin, const float point[3])
{
	float x = point[0] / point[2], y = point[1] / point[2];

	if ((intrin->model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY) ||
		(intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY))
	{

		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		x *= f;
		y *= f;
		float dx = x + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float dy = y + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = dx;
		y = dy;
	}
	if (intrin->model == RS2_DISTORTION_FTHETA)
	{
		float r = sqrtf(x * x + y * y);
		if (r < FLT_EPSILON)
		{
			r = FLT_EPSILON;
		}
		float rd = (float)(1.0f / intrin->coeffs[0] * atan(2 * r * tan(intrin->coeffs[0] / 2.0f)));
		x *= rd / r;
		y *= rd / r;
	}
	if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
	{
		float r = sqrtf(x * x + y * y);
		if (r < FLT_EPSILON)
		{
			r = FLT_EPSILON;
		}
		float theta = atan(r);
		float theta2 = theta * theta;
		float series = 1 + theta2 * (intrin->coeffs[0] + theta2 * (intrin->coeffs[1] + theta2 * (intrin->coeffs[2] + theta2 * intrin->coeffs[3])));
		float rd = theta * series;
		x *= rd / r;
		y *= rd / r;
	}

	pixel[0] = x * intrin->fx + intrin->ppx;
	pixel[1] = y * intrin->fy + intrin->ppy;
}
/* Given pixel coordinates and depth in an image with no distortion or inverse distortion coefficients, compute the corresponding point in 3D space relative to the same camera */
static void rs2_deproject_pixel_to_point(float point[3], const struct rs2_intrinsics* intrin, const float pixel[2], float depth)
{
	assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
	//assert(intrin->model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model

	float x = (pixel[0] - intrin->ppx) / intrin->fx;
	float y = (pixel[1] - intrin->ppy) / intrin->fy;
	if (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
	{
		float r2 = x * x + y * y;
		float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
		float ux = x * f + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
		float uy = y * f + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
		x = ux;
		y = uy;
	}
	if (intrin->model == RS2_DISTORTION_KANNALA_BRANDT4)
	{
		float rd = sqrtf(x * x + y * y);
		if (rd < FLT_EPSILON)
		{
			rd = FLT_EPSILON;
		}

		float theta = rd;
		float theta2 = rd * rd;
		for (int i = 0; i < 4; i++)
		{
			float f = theta * (1 + theta2 * (intrin->coeffs[0] + theta2 * (intrin->coeffs[1] + theta2 * (intrin->coeffs[2] + theta2 * intrin->coeffs[3])))) - rd;
			if (abs(f) < FLT_EPSILON)
			{
				break;
			}
			float df = 1 + theta2 * (3 * intrin->coeffs[0] + theta2 * (5 * intrin->coeffs[1] + theta2 * (7 * intrin->coeffs[2] + 9 * theta2 * intrin->coeffs[3])));
			theta -= f / df;
			theta2 = theta * theta;
		}
		float r = tan(theta);
		x *= r / rd;
		y *= r / rd;
	}
	if (intrin->model == RS2_DISTORTION_FTHETA)
	{
		float rd = sqrtf(x * x + y * y);
		if (rd < FLT_EPSILON)
		{
			rd = FLT_EPSILON;
		}
		float r = (float)(tan(intrin->coeffs[0] * rd) / atan(2 * tan(intrin->coeffs[0] / 2.0f)));
		x *= r / rd;
		y *= r / rd;
	}

	point[0] = depth * x;
	point[1] = depth * y;
	point[2] = depth;
}

#define RandomIterative 0
#define RandomOneshot 1
#define EnsembleRGB 2
#define EnsembleRGBDensity 3
#define PVBCoverage 4
#define Imageto3DCovering 5

class Share_Data
{
public:
	//可变输入参数
	string model_path;
	string ply_file_path;
	string yaml_file_path;
	string name_of_pcd;
	string viewspace_path;
	string instant_ngp_path;
	string orginalviews_path;
	string pvb_path;
	string one12345pp_path;

	int num_of_views;					//一次采样视点个数
	rs2_intrinsics color_intrinsics;
	rs2_intrinsics render_intrinsics;
	double depth_scale;
	double view_space_radius;
	int num_of_thread;

	//运行参数
	bool show;
	int num_of_max_iteration;

	//点云数据
	atomic<int> vaild_clouds;
	vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > clouds;							//点云组
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pcd;
	pcl::PolygonMesh::Ptr mesh_ply;
	int mesh_data_offset;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ground_truth;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_final;
	map<string, double> mp_scale;

	Eigen::Vector3d center_original;
	double scale;

	//八叉地图
	vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> colored_pointclouds;
	shared_ptr<octomap::ColorOcTree> volume_field;

	shared_ptr<octomap::ColorOcTree> octo_model;
	shared_ptr<octomap::ColorOcTree> ground_truth_model;
	shared_ptr<octomap::ColorOcTree> GT_sample;
	double octomap_resolution;
	double ground_truth_resolution;
	double map_size;

	//工作空间与视点空间
	atomic<bool> now_view_space_processed;
	atomic<bool> now_views_infromation_processed;
	atomic<bool> move_on;

	Eigen::Matrix4d now_camera_pose_world;
	Eigen::Vector3d object_center_world;		//物体中心
	double predicted_size;						//物体BBX半径

	int method_of_IG;

	int init_voxels;     //点云voxel个数
	int full_voxels;     //点云voxel个数

	string pre_path;
	string gt_path;
	string save_path;

	vector<vector<double>> pt_sphere;
	double pt_norm;
	double min_z_table;

	int ray_casting_aabb_scale;
	int points_size_cloud;
	int n_steps;
	int eval_n_steps;
	int ensemble_num;
	int evaluate;

	double div_rate_render;
	int num_of_bbx_egde;

	//优化器数据
	int num_of_generation_ensemble;
	set<int> chosen_views; //已经选择的视点
	int num_of_hit_voxels; //所有视点能看到的体素
	vector<vector<octomap::OcTreeKey>> views_voxels; //每个视点能看到的体素
	unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel_id_map; //体素id对应的key
	vector<vector<double>> uncertainty_graph;   //视点i对体素j的不确定度
	vector<vector<double>> distance_graph;   //视点i对视点i的距离
	vector<double> singleview_min_distance;   //视点i到视点的最小距离
	int start_view_id = -1;

	int num_of_min_cover;
	int num_of_min_filter;
	double spatial_uniform_step;
	bool use_gt_mesh;

	bool do_ablation;
	bool eval_mode;
	bool nbv_resume_mode;

	Share_Data(string _config_file_path, string test_name = "", int _num_of_views = -1, int test_method = -1, int _num_of_min_cover = -1)
	{
		yaml_file_path = _config_file_path;
		//读取yaml文件
		cv::FileStorage fs;
		fs.open(yaml_file_path, cv::FileStorage::READ);
		fs["pre_path"] >> pre_path;
		fs["model_path"] >> model_path;
		fs["viewspace_path"] >> viewspace_path;
		fs["instant_ngp_path"] >> instant_ngp_path;
		fs["orginalviews_path"] >> orginalviews_path;
		fs["pvb_path"] >> pvb_path;
		fs["one12345pp_path"] >> one12345pp_path;
		fs["name_of_pcd"] >> name_of_pcd;
		fs["method_of_IG"] >> method_of_IG;
		fs["num_of_thread"] >> num_of_thread;
		fs["octomap_resolution"] >> octomap_resolution;
		fs["ground_truth_resolution"] >> ground_truth_resolution;
		fs["points_size_cloud"] >> points_size_cloud;
		fs["n_steps"] >> n_steps;
		fs["eval_n_steps"] >> eval_n_steps;
		fs["num_of_generation_ensemble"] >> num_of_generation_ensemble;
		fs["num_of_min_cover"] >> num_of_min_cover;
		fs["num_of_min_filter"] >> num_of_min_filter;
		fs["do_ablation"] >> do_ablation;
		fs["eval_mode"] >> eval_mode;
		fs["nbv_resume_mode"] >> nbv_resume_mode;
		fs["spatial_uniform_step"] >> spatial_uniform_step;
		fs["use_gt_mesh"] >> use_gt_mesh;
		fs["evaluate"] >> evaluate;
		fs["ensemble_num"] >> ensemble_num;
		fs["num_of_max_iteration"] >> num_of_max_iteration;
		fs["show"] >> show;
		fs["num_of_views"] >> num_of_views;
		fs["ray_casting_aabb_scale"] >> ray_casting_aabb_scale;
		fs["view_space_radius"] >> view_space_radius;
		fs["div_rate_render"] >> div_rate_render;
		fs["color_width"] >> color_intrinsics.width;
		fs["color_height"] >> color_intrinsics.height;
		fs["color_fx"] >> color_intrinsics.fx;
		fs["color_fy"] >> color_intrinsics.fy;
		fs["color_ppx"] >> color_intrinsics.ppx;
		fs["color_ppy"] >> color_intrinsics.ppy;
		fs["color_model"] >> color_intrinsics.model;
		fs["color_k1"] >> color_intrinsics.coeffs[0];
		fs["color_k2"] >> color_intrinsics.coeffs[1];
		fs["color_k3"] >> color_intrinsics.coeffs[2];
		fs["color_p1"] >> color_intrinsics.coeffs[3];
		fs["color_p2"] >> color_intrinsics.coeffs[4];
		fs["depth_scale"] >> depth_scale;
		fs.release();

		if (test_name != "") name_of_pcd = test_name;
		if (test_method != -1) method_of_IG = test_method;
		if (_num_of_views != -1) num_of_views = _num_of_views;
		if (_num_of_min_cover != -1) num_of_min_cover = _num_of_min_cover;
		
		//读取转换后模型的ply文件
		ply_file_path = model_path + "PLY/";
		mesh_ply.reset(new pcl::PolygonMesh);
		cloud_pcd.reset(new pcl::PointCloud<pcl::PointXYZRGB>);

		ground_truth_model = make_shared<octomap::ColorOcTree>(ground_truth_resolution);
		//ground_truth_model->setProbHit(0.95);	//设置传感器命中率,初始0.7
		//ground_truth_model->setProbMiss(0.05);	//设置传感器失误率，初始0.4
		//ground_truth_model->setClampingThresMax(1.0);	//设置地图节点最大值，初始0.971
		//ground_truth_model->setClampingThresMin(0.0);	//设置地图节点最小值，初始0.1192

		now_camera_pose_world = Eigen::Matrix4d::Identity(4, 4);

		vaild_clouds = 0;
		cloud_final.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud_ground_truth.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
		//path
		gt_path = pre_path + "Coverage_images/";
		save_path = pre_path + "Compare/" ;

		gt_path += name_of_pcd;
		save_path += name_of_pcd;

		cout << "test_method is " << test_method << endl;

		if (test_method != -1) {
			save_path += "_m" + to_string(method_of_IG);

			if (method_of_IG == 5) {
				save_path += "_sc";
				save_path += to_string(num_of_min_cover);
				save_path += "_su";
				std::ostringstream su_str;
				su_str << std::fixed << std::setprecision(2) << spatial_uniform_step; // 设置精度为两位小数
				save_path += su_str.str();
				if (use_gt_mesh) {
					save_path += "_gtm";
				}
			}

			if (method_of_IG == 1) {
				save_path += "_sc";
				save_path += to_string(num_of_min_cover);
			}
		}

		if (method_of_IG == 2) {
			ensemble_num = 2;
			div_rate_render = 10.0;
		}

		if (method_of_IG == 3) {
			ensemble_num = 3;
			div_rate_render = 10.0;
		}

		cout << "gt_path is: " << gt_path << endl;
		cout << "save_path is: " << save_path << endl;

		num_of_bbx_egde = int(round(2.0 / octomap_resolution) + 1e-6);
		cout << "num_of_bbx_egde is: " << num_of_bbx_egde << endl;

		srand(clock());
		//read viewspace
		//ifstream fin_sphere("../view_space_" + to_string(num_of_views) + ".txt");
		ifstream fin_sphere(viewspace_path + to_string(num_of_views) + ".txt");
		pt_sphere.resize(num_of_views);
		for (int i = 0; i < num_of_views; i++) {
			pt_sphere[i].resize(3);
			for (int j = 0; j < 3; j++) {
				fin_sphere >> pt_sphere[i][j];
				//cout << pt_sphere[i][j] << " ??? " << endl;
			}
		}
		cout<< "view space size is: " << pt_sphere.size() << endl;
		Eigen::Vector3d pt0(pt_sphere[0][0], pt_sphere[0][1], pt_sphere[0][2]);
		pt_norm = pt0.norm();
	}

	~Share_Data() {
		//释放内存
		pt_sphere.clear();
		pt_sphere.shrink_to_fit();
		octo_model.reset();
		ground_truth_model.reset();
		GT_sample.reset();
		cloud_pcd->points.clear();
		cloud_pcd->points.shrink_to_fit();
		cloud_pcd.reset();
		mesh_ply->cloud.data.clear();
		mesh_ply->cloud.data.shrink_to_fit();
		mesh_ply.reset();
		cloud_ground_truth->points.clear();
		cloud_ground_truth->points.shrink_to_fit();
		cloud_ground_truth.reset();
		cloud_final->points.clear();
		cloud_final->points.shrink_to_fit();
		cloud_final.reset();
		for (int i = 0; i < clouds.size(); i++) {
			clouds[i]->points.clear();
			clouds[i]->points.shrink_to_fit();
			clouds[i].reset();
		}
		clouds.clear();
		clouds.shrink_to_fit();
	}

	void access_directory(string cd)
	{   //检测多级目录的文件夹是否存在，不存在就创建
		string temp;
		for (int i = 0; i < cd.length(); i++)
			if (cd[i] == '/') {
				if (access(temp.c_str(), 0) != 0) mkdir(temp.c_str());
				temp += cd[i];
			}
			else temp += cd[i];
		if (access(temp.c_str(), 0) != 0) mkdir(temp.c_str());
	}

};

//uility functions
//平方
inline double pow2(double x) {
	return x * x;
}
//投影像素到射线
inline octomap::point3d project_pixel_to_ray_end(float x, float y, rs2_intrinsics& color_intrinsics, Eigen::Matrix4d& now_camera_pose_world, float max_range) {
	float pixel[2] = { float(x),float(y) };
	float point[3];
	rs2_deproject_pixel_to_point(point, &color_intrinsics, pixel, max_range);
	Eigen::Vector4d point_world(point[0], point[1], point[2], 1);
	point_world = now_camera_pose_world * point_world;
	return octomap::point3d(point_world(0), point_world(1), point_world(2));
}
//转换白背景为透明
inline void convertToAlpha(cv::Mat& src, cv::Mat& dst)
{
	cv::cvtColor(src, dst, cv::COLOR_BGR2BGRA);
	for (int y = 0; y < dst.rows; ++y)
	{
		for (int x = 0; x < dst.cols; ++x)
		{
			cv::Vec4b& pixel = dst.at<cv::Vec4b>(y, x);
			if (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255){
				pixel[3] = 0;
			}
		}
	}
}

