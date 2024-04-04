#include <windows.h>
#include <iostream>
#include <cstdio>
#include <thread>
#include <atomic>
#include <chrono>
typedef unsigned unsigned long long pop_t;

using namespace std;

#include "Share_Data.hpp"
#include "View_Space.hpp"
#include <gurobi_c++.h>
#include "json/json.h"

//Virtual_Perception_3D.hpp
class Perception_3D {
public:
	shared_ptr<Share_Data> share_data;
	shared_ptr<octomap::ColorOcTree> ground_truth_model;
	int full_voxels;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
	Eigen::Matrix4d view_pose_world;
	octomap::point3d origin;
	vector<octomap::point3d> end;
	pcl::visualization::PCLVisualizer::Ptr viewer;

	Perception_3D(shared_ptr<Share_Data>& _share_data) {
		share_data = _share_data;
		ground_truth_model = share_data->ground_truth_model;
		full_voxels = share_data->full_voxels;
		view_pose_world = Eigen::Matrix4d::Identity();
		cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
		viewer.reset(new pcl::visualization::PCLVisualizer("Render"));
		viewer->setBackgroundColor(255, 255, 255);
		viewer->initCameraParameters();
		viewer->addPolygonMesh(*share_data->mesh_ply, "mesh_ply");
		//pcl::visualization::Camera cam;
		//viewer->getCameraParameters(cam);
		//cam.window_size[0] = share_data->color_intrinsics.width;
		//cam.window_size[1] = share_data->color_intrinsics.height;
		//viewer->setCameraParameters(cam);
		viewer->setSize(share_data->color_intrinsics.width, share_data->color_intrinsics.height);
	}

	~Perception_3D() {
		viewer->removePolygonMesh("mesh_ply");
		viewer->close();
		viewer.reset();
		end.clear();
		end.shrink_to_fit();
		share_data.reset();
		ground_truth_model.reset();
		cloud->points.clear();
		cloud->points.shrink_to_fit();
		cloud.reset();
		//cout << "Perception_3D: share_data use_count is " << share_data.use_count() << endl;
		//cout << "Perception_3D: ground_truth_model use_count is " << ground_truth_model.use_count() << endl;
		//cout << "Perception_3D: cloud use_count is " << cloud.use_count() << endl;
		//cout << "Perception_3D: viewer use_count is " << viewer.use_count() << endl;
	}

	bool render(View& now_best_view,int id, string path = "") {
		//获取视点位姿
		Eigen::Matrix4d view_pose_world;
		now_best_view.get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
		view_pose_world = (share_data->now_camera_pose_world * now_best_view.pose.inverse()).eval();
		//渲染
		Eigen::Matrix3f intrinsics;
		intrinsics << share_data->color_intrinsics.fx, 0, share_data->color_intrinsics.ppx,
			0, share_data->color_intrinsics.fy, share_data->color_intrinsics.ppy,
			0, 0, 1;
		Eigen::Matrix4f extrinsics = view_pose_world.cast<float>();
		viewer->setCameraParameters(intrinsics, extrinsics);
		viewer->spinOnce(100);
		share_data->access_directory(share_data->gt_path + path);
		viewer->saveScreenshot(share_data->gt_path + path + "/rgb_" + to_string(id) + "_test.png");

		//注意pcl可能缩放了窗口，使用opencv检查图片
		cv::Mat img = cv::imread(share_data->gt_path + path + "/rgb_" + to_string(id) + "_test.png");
		//检查share_data->color_intrinsics.width, share_data->color_intrinsics.height
		if (img.cols != share_data->color_intrinsics.width || img.rows != share_data->color_intrinsics.height) {
			//cout << id << " view-rendered image size is differnet. pick right-bottom block." << "\n";
			img = img(cv::Rect(img.cols - share_data->color_intrinsics.width,
				img.rows - share_data->color_intrinsics.height,
				share_data->color_intrinsics.width,
				share_data->color_intrinsics.height));
		}
		cv::imwrite(share_data->gt_path + path + "/rgb_" + to_string(id) + ".png", img);
		remove((share_data->gt_path + path + "/rgb_" + to_string(id) + "_test.png").c_str());

		//viewer->addCoordinateSystem(1.0);
		//while (!viewer->wasStopped()) {
		//	viewer->spinOnce(100);
		//	this_thread::sleep_for(chrono::milliseconds(100));
		//}

		return true;
	}

	bool precept(View& now_best_view) { 
		double now_time = clock();
		//创建当前成像点云
		cloud->points.clear();
		cloud->points.shrink_to_fit();
		cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud->is_dense = false;
		cloud->points.resize(full_voxels);
		
		//获取视点位姿
		now_best_view.get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
		view_pose_world = (share_data->now_camera_pose_world * now_best_view.pose.inverse()).eval();
		//检查视点的key
		octomap::OcTreeKey key_origin;
		bool key_origin_have = ground_truth_model->coordToKeyChecked(now_best_view.init_pos(0), now_best_view.init_pos(1), now_best_view.init_pos(2), key_origin);
		if (key_origin_have) {
			origin = ground_truth_model->keyToCoord(key_origin);
			//遍历每个体素
			end.resize(full_voxels);
			octomap::ColorOcTree::leaf_iterator it = ground_truth_model->begin_leafs();
			for (int i = 0; i < full_voxels; i++) {
				end[i] = it.getCoordinate();
				it++;
			}
			//ground_truth_model->write(share_data->save_path + "/test_camrea.ot");
			//多线程处理
			vector<thread> precept_process;
			for (int i = 0; i < full_voxels; i+= share_data->num_of_thread) {
				for (int j = 0; j < share_data->num_of_thread && i + j < full_voxels; j++)
					precept_process.push_back(thread(bind(&Perception_3D::precept_thread_process, this, i + j)));
				for (int j = 0; j < share_data->num_of_thread && i + j < full_voxels; j++)
					precept_process[i + j].join();
			}

			//释放内存
			precept_process.clear();
			precept_process.shrink_to_fit();
			end.clear();
			end.shrink_to_fit();
		}
		else {
			cout << "View out of map.check." << endl;
		}
		
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp(new pcl::PointCloud<pcl::PointXYZRGB>);
		temp->is_dense = false;
		temp->points.resize(full_voxels);
		auto ptr = temp->points.begin();
		int vaild_point = 0;
		auto p = cloud->points.begin();
		for (int i = 0; i < cloud->points.size(); i++, p++)
		{
			if ((*p).x == 0 && (*p).y == 0 && (*p).z == 0) continue;
			(*ptr).x = (*p).x;
			(*ptr).y = (*p).y;
			(*ptr).z = (*p).z;
			(*ptr).b = 0;
			(*ptr).g = 0;
			(*ptr).r = 255;
			vaild_point++;
			ptr++;
		}
		temp->width = vaild_point;
		temp->height = 1;
		temp->points.resize(vaild_point);
		
		//记录当前采集点云
		share_data->vaild_clouds++;
		share_data->clouds.push_back(temp);
		//旋转至世界坐标系
		//*share_data->cloud_final += *temp;
		//cout << "virtual cloud get with executed time " << clock() - now_time << " ms." << endl;
		if (share_data->show) {
			//检查读取的偏移
			//viewer->addPointCloud<pcl::PointXYZRGB>(share_data->colored_pointclouds[0], "cloud_gt");
			//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, share_data->points_size_cloud, "cloud_gt");
			//显示成像点云
			pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Camera"));
			viewer1->setBackgroundColor(255, 255, 255);
			viewer1->addCoordinateSystem(1.0);
			viewer1->initCameraParameters();
			viewer1->setSize(share_data->color_intrinsics.width, share_data->color_intrinsics.height);
			Eigen::Matrix3f intrinsics;
			intrinsics << share_data->color_intrinsics.fx, 0, share_data->color_intrinsics.ppx,
				0, share_data->color_intrinsics.fy, share_data->color_intrinsics.ppy,
				0, 0, 1;
			Eigen::Matrix4f extrinsics = view_pose_world.cast<float>();
			viewer1->setCameraParameters(intrinsics, extrinsics);
			viewer1->addPointCloud<pcl::PointXYZRGB>(temp, "cloud");
			viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
			//检查成像贴合度
			viewer1->addPointCloud<pcl::PointXYZRGB>(share_data->cloud_ground_truth, "cloud_gt");
			viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, share_data->points_size_cloud, "cloud_gt");
			Eigen::Vector4d X(0.5, 0, 0, 1);
			Eigen::Vector4d Y(0, 0.5, 0, 1);
			Eigen::Vector4d Z(0, 0, 0.5, 1);
			Eigen::Vector4d O(0, 0, 0, 1);
			X = view_pose_world * X;
			Y = view_pose_world * Y;
			Z = view_pose_world * Z;
			O = view_pose_world * O;
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
			viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
			viewer1->spinOnce(100);
			viewer1->saveScreenshot(share_data->save_path + "/cloud.png");
			while (!viewer1->wasStopped())
			{
				viewer1->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
			viewer1->removeAllCoordinateSystems();
			viewer1->removeAllPointClouds();
			viewer1->removeAllShapes();
			viewer1->close();
			viewer1.reset();

			//显示了投影图像是180转置
			cv::Mat color = cv::Mat(share_data->color_intrinsics.height, share_data->color_intrinsics.width, CV_8UC3);
			for (int y = 0; y < color.rows; ++y) {
				for (int x = 0; x < color.cols; ++x) {
					cv::Vec3b& pixel = color.at<cv::Vec3b>(y, x);
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
				}
			}
			for (int i = 0; i < cloud->points.size(); i++) {
				Eigen::Vector4d end_3d(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1);
				auto vertex = view_pose_world.inverse() * end_3d;
				float point_3d[3] = { vertex(0), vertex(1),vertex(2) };
				float pixel[2];
				rs2_project_point_to_pixel(pixel, &share_data->color_intrinsics, point_3d);
				color.at<cv::Vec3b>(pixel[1], pixel[0])[0] = cloud->points[i].b;
				color.at<cv::Vec3b>(pixel[1], pixel[0])[1] = cloud->points[i].g;
				color.at<cv::Vec3b>(pixel[1], pixel[0])[2] = cloud->points[i].r;
			}
			cv::imwrite(share_data->save_path + "/color.png", color);
			cv::imshow("color", color);
			cv::waitKey(0);
		}

		return true;
	}

	void precept_thread_process(int i) {
		//num++;
		pcl::PointXYZRGB point;
		point.x = 0; point.y = 0; point.z = 0;
		//投影检测是否在成像范围内
		Eigen::Vector4d end_3d(end[i].x(), end[i].y(), end[i].z(), 1);
		Eigen::Vector4d vertex = view_pose_world.inverse() * end_3d;
		float point_3d[3] = { vertex(0), vertex(1),vertex(2) };
		float pixel[2];
		rs2_project_point_to_pixel(pixel, &share_data->color_intrinsics, point_3d);
		if (pixel[0] < 0 || pixel[0]>share_data->color_intrinsics.width || pixel[1] < 0 || pixel[1]>share_data->color_intrinsics.height) {
			cloud->points[i] = point;
			return;
		}
		//反向投影找到终点
		octomap::point3d end = project_pixel_to_ray_end(pixel[0], pixel[1], share_data->color_intrinsics, view_pose_world, 2.0);
		octomap::OcTreeKey key_end;
		octomap::point3d direction = end - origin;
		octomap::point3d end_point;
		//越过未知区域，找到终点
		bool found_end_point = ground_truth_model->castRay(origin, direction, end_point, true, 4.0);
		if (!found_end_point) {//未找到终点，无观测数据
			cloud->points[i] = point;
			return;
		}
		if (end_point == origin) {
			cout << "view in the object. check!" << endl;
			cloud->points[i] = point;
			return;
		}
		//检查一下末端是否在地图限制范围内
		bool key_end_have = ground_truth_model->coordToKeyChecked(end_point, key_end);
		if (key_end_have) {
			octomap::ColorOcTreeNode* node = ground_truth_model->search(key_end);
			if (node != NULL) {
				octomap::ColorOcTreeNode::Color color = node->getColor();
				point.x = end_point.x();
				point.y = end_point.y();
				point.z = end_point.z();
				point.r = color.r;
				point.g = color.g;
				point.b = color.b;
				node = NULL;
			}
		}
		cloud->points[i] = point;
	}

};

//Global_Path_Planner.hpp
/* Solve a traveling salesman problem on a randomly generated set of
	points using lazy constraints.   The base MIP model only includes
	'degree-2' constraints, requiring each node to have exactly
	two incident edges.  Solutions to this model may contain subtours -
	tours that don't visit every node.  The lazy constraint callback
	adds new constraints to cut them off. */
// Given an integer-feasible solution 'sol', find the smallest
// sub-tour.  Result is returned in 'tour', and length is
// returned in 'tourlenP'.
void findsubtour(int n, double** sol, int* tourlenP, int* tour) {
	{
		bool* seen = new bool[n];
		int bestind, bestlen;
		int i, node, len, start;

		for (i = 0; i < n; i++)
			seen[i] = false;

		start = 0;
		bestlen = n + 1;
		bestind = -1;
		node = 0;
		while (start < n) {
			for (node = 0; node < n; node++)
				if (!seen[node])
					break;
			if (node == n)
				break;
			for (len = 0; len < n; len++) {
				tour[start + len] = node;
				seen[node] = true;
				for (i = 0; i < n; i++) {
					if (sol[node][i] > 0.5 && !seen[i]) {
						node = i;
						break;
					}
				}
				if (i == n) {
					len++;
					if (len < bestlen) {
						bestlen = len;
						bestind = start;
					}
					start += len;
					break;
				}
			}
		}

		for (i = 0; i < bestlen; i++)
			tour[i] = tour[bestind + i];
		*tourlenP = bestlen;

		delete[] seen;
	}
}
// Subtour elimination callback.  Whenever a feasible solution is found,
// find the smallest subtour, and add a subtour elimination constraint
// if the tour doesn't visit every node.
class subtourelim : public GRBCallback
{
public:
	GRBVar** vars;
	int n;
	subtourelim(GRBVar** xvars, int xn) {
		vars = xvars;
		n = xn;
	}
protected:
	void callback() {
		try {
			if (where == GRB_CB_MIPSOL) {
				// Found an integer feasible solution - does it visit every node?
				double** x = new double* [n];
				int* tour = new int[n];
				int i, j, len;
				for (i = 0; i < n; i++)
					x[i] = getSolution(vars[i], n);

				findsubtour(n, x, &len, tour);

				if (len < n) {
					// Add subtour elimination constraint
					GRBLinExpr expr = 0;
					for (i = 0; i < len; i++)
						for (j = i + 1; j < len; j++)
							expr += vars[tour[i]][tour[j]];
					addLazy(expr <= len - 1);
				}

				for (i = 0; i < n; i++)
					delete[] x[i];
				delete[] x;
				delete[] tour;
			}
		}
		catch (GRBException e) {
			cout << "Error number: " << e.getErrorCode() << endl;
			cout << e.getMessage() << endl;
		}
		catch (...) {
			cout << "Error during callback" << endl;
		}
	}
};
class Global_Path_Planner {
public:
	shared_ptr<Share_Data> share_data;
	int now_view_id;
	int end_view_id;
	bool solved;
	int n;
	map<int, int>* view_id_in;
	map<int, int>* view_id_out;
	vector<vector<double>> graph;
	double total_shortest;
	vector<int> global_path;
	GRBEnv* env = NULL;
	GRBVar** vars = NULL;
	GRBModel* model = NULL;
	subtourelim* cb = NULL;
	
	Global_Path_Planner(shared_ptr<Share_Data> _share_data, vector<View>& views, vector<int>& view_set_label, int _now_view_id, int _end_view_id = -1) {
		share_data = _share_data;
		now_view_id = _now_view_id;
		end_view_id = _end_view_id;
		solved = false;
		total_shortest = -1;
		//构造下标映射
		view_id_in = new map<int, int>();
		view_id_out = new map<int, int>();
		for (int i = 0; i < view_set_label.size(); i++) {
			(*view_id_in)[view_set_label[i]] = i;
			(*view_id_out)[i] = view_set_label[i];
		}
		(*view_id_in)[views.size()] = view_set_label.size(); //注意复制节点应该是和视点空间个数相关，映射到所需视点个数
		(*view_id_out)[view_set_label.size()] = views.size(); 
		//节点数为原始个数+起点的复制节点
		n = view_set_label.size() + 1;
		//local path 完全无向图
		graph.resize(n);
		for (int i = 0; i < n; i++)
			graph[i].resize(n);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				if (i == n - 1 || j == n - 1) {
					//如果是起点的复制节点，距离为0
					graph[i][j] = 0.0;
				}
				else {
					//交换id
					int u = (*view_id_out)[i];
					int v = (*view_id_out)[j];
					//求两点路径
					pair<int, double> local_path = get_local_path(views[u].init_pos.eval(), views[v].init_pos.eval(), (share_data->object_center_world + Eigen::Vector3d(1e-10, 1e-10, 1e-10)).eval(), share_data->predicted_size);
					if (local_path.first < 0) {
						cout << "local path not found." << endl;
						graph[i][j] = 1e10;
					}
					else graph[i][j] = local_path.second;
				}
				//cout << "graph[" << i << "][" << j << "] = " << graph[i][j] << endl;
			}
		//创建Gurobi的TSP优化器
		vars = new GRBVar * [n];
		for (int i = 0; i < n; i++)
			vars[i] = new GRBVar[n];
		env = new GRBEnv();
		model = new GRBModel(*env);
		//cout << "Gurobi model created." << endl;
		// Must set LazyConstraints parameter when using lazy constraints
		model->set(GRB_IntParam_LazyConstraints, 1);
		//cout << "Gurobi set LazyConstraints." << endl;
		// Create binary decision variables
		for (int i = 0; i < n; i++) {
			for (int j = 0; j <= i; j++) {
				vars[i][j] = model->addVar(0.0, 1.0, graph[i][j], GRB_BINARY, "x_" + to_string(i) + "_" + to_string(j));
				vars[j][i] = vars[i][j];
			}
		}
		//cout << "Gurobi addVar complete." << endl;
		// Degree-2 constraints
		for (int i = 0; i < n; i++) {
			GRBLinExpr expr = 0;
			for (int j = 0; j < n; j++)
				expr += vars[i][j];
			model->addConstr(expr == 2, "deg2_" + to_string(i));
		}
		//cout << "Gurobi add Degree-2 Constr complete." << endl;
		// Forbid edge from node back to itself
		for (int i = 0; i < n; i++)
			vars[i][i].set(GRB_DoubleAttr_UB, 0);
		//cout << "Gurobi set Forbid edge complete." << endl;
		// Make copy node linked to starting node
		vars[n - 1][(*view_id_in)[now_view_id]].set(GRB_DoubleAttr_LB, 1);
		// 默认不设置终点，多解只返回一个
		if(end_view_id != -1) vars[(*view_id_in)[end_view_id]][n - 1].set(GRB_DoubleAttr_LB, 1);
		//cout << "Gurobi set Make copy node complete." << endl;
		// Set callback function
		cb = new subtourelim(vars, n);
		model->setCallback(cb);
		//cout << "Gurobi set callback function complete." << endl;
		cout << "Global_Path_Planner inited." << endl;
	}

	~Global_Path_Planner() {
		graph.clear();
		graph.shrink_to_fit();
		global_path.clear();
		global_path.shrink_to_fit();
		for (int i = 0; i < n; i++)
			delete[] vars[i];
		delete[] vars;
		delete env;
		delete model;
		delete cb;
	}

	double solve() {
		double now_time = clock();
		// Optimize model
		model->optimize();
		// Extract solution
		if (model->get(GRB_IntAttr_SolCount) > 0) {
			solved = true;
			total_shortest = 0.0;
			double** sol = new double* [n];
			for (int i = 0; i < n; i++)
				sol[i] = model->get(GRB_DoubleAttr_X, vars[i], n);

			int* tour = new int[n];
			int len;

			findsubtour(n, sol, &len, tour);
			assert(len == n);

			//cout << "Tour: ";
			for (int i = 0; i < len; i++) {
				global_path.push_back(tour[i]);
				if (i != len - 1) {
					total_shortest += graph[tour[i]][tour[i + 1]];
				}
				else {
					total_shortest += graph[tour[i]][tour[0]];
				}
				//cout << tour[i] << " ";
			}
			//cout << endl;

			for (int i = 0; i < n; i++)
				delete[] sol[i];
			delete[] sol;
			delete[] tour;
		}
		else {
			cout << "No solution found" << endl;
		}
		double cost_time = clock() - now_time;
		cout << "Global Path length " << total_shortest << " getted with executed time " << cost_time << " ms." << endl;
		return total_shortest;
	}

	vector<int> get_path_id_set() {
		if (!solved) cout << "call solve() first" << endl;
		cout << "Node ids on global_path form start to end are: ";
		vector<int> ans = global_path;
		//调准顺序把复制的起点置于末尾
		int copy_node_id = -1;
		for (int i = 0; i < ans.size(); i++) {
			if (ans[i] == n - 1) {
				copy_node_id = i;
				break;
			}
		}
		if (copy_node_id == -1) {
			cout << "copy_node_id == -1" << endl;
		}
		for (int i = 0; i < copy_node_id; i++) {
			ans.push_back(ans[0]);
			ans.erase(ans.begin());
		}
		//删除复制的起点
		for (int i = 0; i < ans.size(); i++) {
			if (ans[i] == n - 1) {
				ans.erase(ans.begin() + i);
				break;
			}
		}
		//如果起点是第一个就不动，是最后一个就反转
		if (ans[0] != (*view_id_in)[now_view_id]) {
			reverse(ans.begin(), ans.end());
		}
		for (int i = 0; i < ans.size(); i++) {
			ans[i] = (*view_id_out)[ans[i]];
			cout << ans[i] << " ";
		}
		cout << endl;
		//删除出发点
		//ans.erase(ans.begin());
		return ans;
	}
};

//views_voxels_set_covering.hpp
class views_voxels_set_covering {
public:
	Share_Data* share_data;
	View_Space* view_space;
	Perception_3D* perception_3d;
	double cost_rate;

	set<int> chosen_views;
	int num_considered_voxels;
	vector<set<int>> view_oberseved_voxel_set;
	vector<int> voxel_num_count;
	vector<int> num_of_less_than_voxels;
	GRBEnv* env;
	GRBModel* model;
	vector<GRBVar> x;
	GRBLinExpr obj;

	views_voxels_set_covering(Share_Data* _share_data, View_Space* _view_space, Perception_3D* _perception_3d) {
		share_data = _share_data;
		view_space = _view_space;
		perception_3d = _perception_3d;
		cost_rate = 0.0;
		double now_time = clock();
		chosen_views.insert(share_data->start_view_id);
		//对于每个视点进行光线投射,建立体素id和key的映射
		share_data->voxel_id_map = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
		share_data->views_voxels.resize(share_data->num_of_views);
		share_data->num_of_hit_voxels = 0;
		//对每个模型进行投射
		cout << "num_of_generation_ensemble is " << share_data->num_of_generation_ensemble << endl;
		for (int ensemble_id = 0; ensemble_id < share_data->num_of_generation_ensemble; ensemble_id++) {
			share_data->volume_field = make_shared<octomap::ColorOcTree>(share_data->octomap_resolution);
			//把采样的点云插入volume_field
			for (int i = 0; i < share_data->colored_pointclouds[ensemble_id]->points.size(); i++) {
				double x = share_data->colored_pointclouds[ensemble_id]->points[i].x;
				double y = share_data->colored_pointclouds[ensemble_id]->points[i].y;
				double z = share_data->colored_pointclouds[ensemble_id]->points[i].z;
				octomap::OcTreeKey key;  bool key_have = share_data->volume_field->coordToKeyChecked(octomap::point3d(x, y, z), key);
				if (!key_have) cout << "key_have false" << endl;
				octomap::point3d voxel_center = share_data->volume_field->keyToCoord(key);
				share_data->volume_field->setNodeValue(key, share_data->volume_field->getProbHitLog(), true);
				share_data->volume_field->integrateNodeColor(key, share_data->colored_pointclouds[ensemble_id]->points[i].r, share_data->colored_pointclouds[ensemble_id]->points[i].g, share_data->colored_pointclouds[ensemble_id]->points[i].b);
			}
			share_data->volume_field->updateInnerOccupancy();
			//share_data->volume_field->write(share_data->one12345pp_path + share_data->name_of_pcd + "/volume_field_scop_" + to_string(ensemble_id) + ".ot");
			//重新初始化模拟相机
			perception_3d->ground_truth_model = share_data->volume_field;
			perception_3d->full_voxels = 0;
			for (octomap::ColorOcTree::leaf_iterator it = perception_3d->ground_truth_model->begin_leafs(), end = perception_3d->ground_truth_model->end_leafs(); it != end; ++it) {
				perception_3d->full_voxels++;
			}
			cout << "full_voxels: " << perception_3d->full_voxels << endl;
			//遍历views
			cout << "start ray casting for generated mesh " << ensemble_id << " ..." << endl;
			for (int i = 0; i < share_data->num_of_views; i++) {
				//cout << "raycating view " << i << endl;
				perception_3d->precept(view_space->views[i]);
				//get voxel map
				int num = 0;
				unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>* voxel = new unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash>();
				for (int j = 0; j < share_data->clouds[i]->points.size(); j++) {
					octomap::OcTreeKey key = share_data->volume_field->coordToKey(share_data->clouds[i]->points[j].x, share_data->clouds[i]->points[j].y, share_data->clouds[i]->points[j].z);
					//找到了体素，建立不重复映射
					if (share_data->voxel_id_map->find(key) == share_data->voxel_id_map->end()) {
						(*share_data->voxel_id_map)[key] = share_data->num_of_hit_voxels++;
					}
					//记录像素对应的体素
					share_data->views_voxels[i].push_back(key);
				}
			}
			//输出时间
			cout << "time for ray casting: " << (clock() - now_time) / CLOCKS_PER_SEC << "s" << endl;
			now_time = clock();
			//清除之前的点云
			share_data->clouds.clear();
		}
		//检查视点可以看到哪些体素
		view_oberseved_voxel_set.resize(share_data->num_of_views);
		voxel_num_count.resize(share_data->num_of_hit_voxels);
		for (int j = 0; j < share_data->num_of_hit_voxels; j++) {
			voxel_num_count[j] = 0;
		}
		for (int i = 0; i < share_data->num_of_views; i++) {
			vector<bool> num_updated;
			for (int j = 0; j < share_data->num_of_hit_voxels; j++) {
				num_updated.push_back(false);
			}
			for (int j = 0; j < share_data->views_voxels[i].size(); j++) {
				int voxel_id = (*share_data->voxel_id_map)[share_data->views_voxels[i][j]];
				view_oberseved_voxel_set[i].insert(voxel_id);
				if (!num_updated[voxel_id]) {
					voxel_num_count[voxel_id]++;
					num_updated[voxel_id] = true;
				}
			}
		}
		// 计算距离约束
		double total_min_distance = 1e100;
		share_data->singleview_min_distance.resize(share_data->num_of_views);
		share_data->distance_graph.resize(share_data->num_of_views);
		for (int i = 0; i < share_data->num_of_views; i++) {
			share_data->distance_graph[i].resize(share_data->num_of_views);
			share_data->singleview_min_distance[i] = 1e100;
			for (int j = 0; j < share_data->num_of_views; j++) {
				share_data->distance_graph[i][j] = (view_space->views[i].init_pos - view_space->views[j].init_pos).norm();
				if (i != j) share_data->singleview_min_distance[i] = min(share_data->singleview_min_distance[i], share_data->distance_graph[i][j]);
			}
			total_min_distance = min(total_min_distance, share_data->singleview_min_distance[i]);
		}
		cout << "total_min_distance: " << total_min_distance << endl;
	}

	void solve_linear() {
		double now_time = clock();
		if (share_data->spatial_uniform_step) {
			cout << "share_data->spatial_uniform_step > 0 solve with spatial constrains by step " << share_data->spatial_uniform_step << endl;
			ofstream out_iter(share_data->save_path + "/spatial_uniform.txt");
			//如果需要spatial_uniform
			cost_rate = 1.0;
			while (true) {
				//求解一次
				solve_once();
				//如果无解则退出
				int status = model->get(GRB_IntAttr_Status);
				if (!(status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL)) {
					break;
				}
				//保存当前的情况
				int num_of_selected_views = 0;
				for (int i = 0; i < share_data->num_of_views; i++)
					if (x[i].get(GRB_DoubleAttr_X) == 1.0) num_of_selected_views++;
				out_iter << cost_rate << "\t" << num_of_selected_views << "\t";
				for (int i = 0; i < share_data->num_of_views; i++)
					if (x[i].get(GRB_DoubleAttr_X) == 1.0) out_iter << i << " ";
				out_iter << "\n";
				//如果有解则继续增加step
				cost_rate += share_data->spatial_uniform_step;
			}
			//退回一步到最优解
			cost_rate -= share_data->spatial_uniform_step;
			solve_once();
			out_iter.close();
		}
		else {
			cout << "share_data->spatial_uniform_step < 0 solve without spatial constrains." << endl;
			//只求解一次
			cost_rate = 0.0;
			solve_once();
		}
		cout << "Integer linear program iterated with executed time " << clock() - now_time << " ms." << endl;
	}

	void solve() {
		double now_time = clock();
		if (share_data->spatial_uniform_step) {
			//如果需要spatial_uniform
			cout << "share_data->spatial_uniform_step > 0 solve with spatial constrains by step " << share_data->spatial_uniform_step << endl;
			ofstream out_iter(share_data->save_path + "/spatial_uniform.txt");
			//构建二分查找数组
			vector<double> beta_values;
			beta_values.push_back(0.0);
			for (double beta = 1.0; beta <= 10.0; beta += share_data->spatial_uniform_step) {
				beta_values.push_back(beta);
			}
			int left = 0, right = beta_values.size() - 1;
			while (left < right) {
				int mid = (left + right) / 2;
				cost_rate = beta_values[mid];
				//求解一次
				solve_once();
				int status = model->get(GRB_IntAttr_Status);
				if (status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL) {
					left = mid + 1;
					//保存当前的情况
					int num_of_selected_views = 0;
					for (int i = 0; i < share_data->num_of_views; i++)
						if (x[i].get(GRB_DoubleAttr_X) == 1.0) num_of_selected_views++;
					out_iter << cost_rate << "\t" << num_of_selected_views << "\t";
					for (int i = 0; i < share_data->num_of_views; i++)
						if (x[i].get(GRB_DoubleAttr_X) == 1.0) out_iter << i << " ";
					out_iter << "\n";
				}
				else {
					right = mid;
				}
			}
			int ans = left - 1;
			cost_rate = beta_values[ans];
			solve_once();
			out_iter.close();
		}
		else {
			cout << "share_data->spatial_uniform_step < 0 solve without spatial constrains." << endl;
			//只求解一次
			cost_rate = 0.0;
			solve_once();
		}
		cout << "Integer linear program iterated with executed time " << clock() - now_time << " ms." << endl;
	}

	void solve_once() {
		//建立对应的线性规划求解器
		double now_time = clock();
		env = new GRBEnv();
		model = new GRBModel(*env);
		x.resize(share_data->num_of_views);
		// Create variables
		for (int i = 0; i < share_data->num_of_views; i++)
			x[i] = model->addVar(0.0, 1.0, 0.0, GRB_BINARY, "x" + to_string(i));
		for (auto it = chosen_views.begin(); it != chosen_views.end(); it++) {
			//已经选择的视点
			x[*it].set(GRB_DoubleAttr_LB, 1.0);
			x[*it].set(GRB_DoubleAttr_UB, 1.0);
		}
		// Set objective : \sum_{s\in S} x_s
		for (int i = 0; i < share_data->num_of_views; i++)
			obj += x[i];
		model->setObjective(obj, GRB_MINIMIZE);
		// Add linear constraint: \sum_{S:e\in S} x_s\geq1
		num_considered_voxels = 0;
		for (int j = 0; j < share_data->num_of_hit_voxels; j++)
		{
			if (voxel_num_count[j] <= share_data->num_of_min_filter) {
				continue;
			}
			GRBLinExpr subject_of_voxel;
			for (int i = 0; i < share_data->num_of_views; i++) {
				if (view_oberseved_voxel_set[i].find(j) != view_oberseved_voxel_set[i].end()) {
					subject_of_voxel += x[i];
				}
			}
			//model->addConstr(subject_of_voxel >= 1, "c" + to_string(j));
			model->addConstr(subject_of_voxel >= share_data->num_of_min_cover, "c" + to_string(j));
			num_considered_voxels++;
		}
		// 只考虑需要plan的视点空间
		cout << "cost_rate: " << cost_rate << endl;
		for (int i = 0; i < share_data->num_of_views; i++) {
			for (int j = i + 1; j < share_data->num_of_views; j++) {
				//如果两个视点之间的距离大于share_data->cost_rate倍的最小距离，那么不需要约束，也就是只考虑最近邻域的点
				if (share_data->distance_graph[i][j] <= share_data->singleview_min_distance[i] * cost_rate || share_data->distance_graph[j][i] <= share_data->singleview_min_distance[j] * cost_rate) {
					//负约束，确保这个范围内的两个视点无法被同时选择
					GRBLinExpr subject_distance = x[i] + x[j] - 1;
					model->addConstr(subject_distance <= 0, "subject_distance" + to_string(i) + "," + to_string(j));
				}
			}
		}
		model->set("TimeLimit", "600");
		cout << "Integer linear program formulated with executed time " << clock() - now_time << " ms." << endl;
		// Optimize model
		model->optimize();
	}

	~views_voxels_set_covering() {
		delete model;
		delete env;
	}

	vector<int> get_view_id_set() {
		vector<int> ans;
		for (int i = 0; i < share_data->num_of_views; i++)
			if (x[i].get(GRB_DoubleAttr_X) == 1.0) ans.push_back(i);
		return ans;
	}

};

//View_Planning_Simulator.hpp
class View_Planning_Simulator
{
public:
	shared_ptr<Share_Data> share_data;
	shared_ptr<View_Space> view_space;
	shared_ptr<Perception_3D> percept;
	vector<View> init_views;

	~View_Planning_Simulator() {
		share_data.reset();
		view_space.reset();
		percept.reset();
		init_views.clear();
		init_views.shrink_to_fit();
	}

	View_Planning_Simulator(shared_ptr<Share_Data>& _share_data) {
		share_data = _share_data;
		//读取mesh
		if (pcl::io::loadPolygonFilePLY(share_data->ply_file_path + share_data->name_of_pcd + ".ply", *share_data->mesh_ply) == -1) {
			cout << "Mesh not available. Please check if the file (or path) exists." << endl;
		}
		share_data->mesh_data_offset = share_data->mesh_ply->cloud.data.size() / share_data->mesh_ply->cloud.width / share_data->mesh_ply->cloud.height;
		cout << "mesh field offset is " << share_data->mesh_data_offset << endl;
		//从mesh中获取点云，转为pcl::PointXYZRGB
		pcl::fromPCLPointCloud2(share_data->mesh_ply->cloud, *share_data->cloud_pcd);
		cout << "Mesh and points has been loaded successfully." << endl;
		//GT cloud
		share_data->cloud_ground_truth->is_dense = false;
		share_data->cloud_ground_truth->points.resize(share_data->cloud_pcd->points.size());
		share_data->cloud_ground_truth->width = share_data->cloud_pcd->points.size();
		share_data->cloud_ground_truth->height = 1;
		auto ptr = share_data->cloud_ground_truth->points.begin();
		auto p = share_data->cloud_pcd->points.begin();
		//检查物体中心
		vector<Eigen::Vector3d> points;
		for (auto& ptr : share_data->cloud_pcd->points) {
			//cout<< "ptr: " << ptr.x << " " << ptr.y << " " << ptr.z << endl;
			Eigen::Vector3d pt(ptr.x, ptr.y, ptr.z);
			points.push_back(pt);
		}
		Eigen::Vector3d object_center_world = Eigen::Vector3d(0, 0, 0);
		//计算点云质心
		for (auto& ptr : points) {
			object_center_world(0) += ptr(0);
			object_center_world(1) += ptr(1);
			object_center_world(2) += ptr(2);
		}
		object_center_world(0) /= points.size();
		object_center_world(1) /= points.size();
		object_center_world(2) /= points.size();
		share_data->center_original = object_center_world;
		cout<< "center original is " << object_center_world(0) << " " << object_center_world(1) << " " << object_center_world(2) << endl;
		//平移点云使得重心为0,0,0
		for (auto& ptr : share_data->cloud_pcd->points) {
			ptr.x = (ptr.x - object_center_world(0));
			ptr.y = (ptr.y - object_center_world(1));
			ptr.z = (ptr.z - object_center_world(2));
		}
		for (int i = 0; i < share_data->mesh_ply->cloud.data.size(); i += share_data->mesh_data_offset) {
			int arrayPosX = i + share_data->mesh_ply->cloud.fields[0].offset;
			int arrayPosY = i + share_data->mesh_ply->cloud.fields[1].offset;
			int arrayPosZ = i + share_data->mesh_ply->cloud.fields[2].offset;
			float X = 0.0;	float Y = 0.0;	float Z = 0.0;
			memcpy(&X, &share_data->mesh_ply->cloud.data[arrayPosX], sizeof(float));
			memcpy(&Y, &share_data->mesh_ply->cloud.data[arrayPosY], sizeof(float));
			memcpy(&Z, &share_data->mesh_ply->cloud.data[arrayPosZ], sizeof(float));
			X = float(X - object_center_world(0));
			Y = float(Y - object_center_world(1));
			Z = float(Z - object_center_world(2));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosX], &X, sizeof(float));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosY], &Y, sizeof(float));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosZ], &Z, sizeof(float));
		}
		//再次计算中心
		points.clear();
		points.shrink_to_fit();
		for (auto& ptr : share_data->cloud_pcd->points) {
			Eigen::Vector3d pt(ptr.x, ptr.y, ptr.z);
			points.push_back(pt);
		}
		object_center_world = Eigen::Vector3d(0, 0, 0);
		//计算点云质心
		for (auto& ptr : points) {
			object_center_world(0) += ptr(0);
			object_center_world(1) += ptr(1);
			object_center_world(2) += ptr(2);
		}
		object_center_world(0) /= points.size();
		object_center_world(1) /= points.size();
		object_center_world(2) /= points.size();
		if (object_center_world.norm() > 1e-6) {
			cout << "error with move to centre." << endl;
		}
		//计算最远点
		double predicted_size = 0.0;
		for (auto& ptr : points) {
			predicted_size = max(predicted_size, (object_center_world - ptr).norm());
		}
		double scale = 1.0;
		//固定大小为1.0m
		scale = 1.0 / predicted_size;
		share_data->scale = scale;
		cout << "object " << share_data->name_of_pcd << " change scale " << predicted_size << " to about " << 1.0 << " m." << endl;
		//释放内存
		points.clear();
		points.shrink_to_fit();
		////动态分辨率
		//double predicted_octomap_resolution = 2.0 / 64;
		//cout << "choose octomap_resolution: " << predicted_octomap_resolution << " m." << endl;
		//share_data->octomap_resolution = predicted_octomap_resolution;
		cout << "octomap_resolution is now : " << share_data->octomap_resolution << " m." << endl;
		cout << "bbx egde num is now : " << share_data->num_of_bbx_egde << " m." << endl;
		share_data->octo_model = make_shared<octomap::ColorOcTree>(share_data->octomap_resolution);
		share_data->GT_sample = make_shared<octomap::ColorOcTree>(share_data->octomap_resolution);
		//测试BBX尺寸
		for (int i = 0; i < 32; i++)
			for (int j = 0; j < 32; j++)
				for (int k = 0; k < 32; k++)
				{
					double x = object_center_world(0) * scale - scale * predicted_size + share_data->octomap_resolution * i;
					double y = object_center_world(1) * scale - scale * predicted_size + share_data->octomap_resolution * j;
					double z = object_center_world(2) * scale - scale * predicted_size + share_data->octomap_resolution * k;
					share_data->GT_sample->setNodeValue(x, y, z, share_data->GT_sample->getProbMissLog(), true); //初始化概率0
					//cout << x << " " << y << " " << z << endl;
				}

		//转换mesh
		for (int i = 0; i < share_data->mesh_ply->cloud.data.size(); i += share_data->mesh_data_offset) {
			int arrayPosX = i + share_data->mesh_ply->cloud.fields[0].offset;
			int arrayPosY = i + share_data->mesh_ply->cloud.fields[1].offset;
			int arrayPosZ = i + share_data->mesh_ply->cloud.fields[2].offset;
			float X = 0.0;	float Y = 0.0;	float Z = 0.0;
			memcpy(&X, &share_data->mesh_ply->cloud.data[arrayPosX], sizeof(float));
			memcpy(&Y, &share_data->mesh_ply->cloud.data[arrayPosY], sizeof(float));
			memcpy(&Z, &share_data->mesh_ply->cloud.data[arrayPosZ], sizeof(float));
			X = float(X * scale);
			Y = float(Y * scale);
			Z = float(Z * scale);
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosX], &X, sizeof(float));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosY], &Y, sizeof(float));
			memcpy(&share_data->mesh_ply->cloud.data[arrayPosZ], &Z, sizeof(float));
		}

		//转换点云
		//double min_z = 0;
		double min_z = object_center_world(2) * scale;
		for (int i = 0; i < share_data->cloud_pcd->points.size(); i++, p++)
		{
			//RGB点云
			(*ptr).x = (*p).x * scale;
			(*ptr).y = (*p).y * scale;
			(*ptr).z = (*p).z * scale;
			(*ptr).r = (*p).r;
			(*ptr).g = (*p).g;
			(*ptr).b = (*p).b;
			//GT OctoMap插入点云
			octomap::OcTreeKey key;  bool key_have = share_data->ground_truth_model->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key);
			if (key_have) {
				octomap::ColorOcTreeNode* voxel = share_data->ground_truth_model->search(key);
				if (voxel == NULL) {
					share_data->ground_truth_model->setNodeValue(key, share_data->ground_truth_model->getProbHitLog(), true);
					share_data->ground_truth_model->integrateNodeColor(key, (*ptr).r, (*ptr).g, (*ptr).b);
				}
				voxel = NULL;
			}
			min_z = min(min_z, (double)(*ptr).z);
			//GT_sample OctoMap插入点云
			octomap::OcTreeKey key_sp;  bool key_have_sp = share_data->GT_sample->coordToKeyChecked(octomap::point3d((*ptr).x, (*ptr).y, (*ptr).z), key_sp);
			if (key_have_sp) {
				octomap::ColorOcTreeNode* voxel_sp = share_data->GT_sample->search(key_sp);
				//if (voxel_sp == NULL) {
					share_data->GT_sample->setNodeValue(key_sp, share_data->GT_sample->getProbHitLog(), true);
					share_data->GT_sample->integrateNodeColor(key_sp, 255, 0, 0);
				//}
				voxel_sp = NULL;
			}
			ptr++;
		}
		//记录桌面
		share_data->min_z_table = min_z - share_data->ground_truth_resolution;
		////保存点云share_data->cloud_ground_truth 
		//pcl::io::savePCDFile(share_data->one12345pp_path + share_data->name_of_pcd + "/cloud_ground_truth_normilzed.pcd", *share_data->cloud_ground_truth);
		////保存mesh
		//pcl::io::savePLYFile(share_data->one12345pp_path + share_data->name_of_pcd + "/mesh_normilzed.ply", *share_data->mesh_ply);
		//share_data->access_directory(share_data->save_path);
		share_data->ground_truth_model->updateInnerOccupancy();
		//share_data->ground_truth_model->write(share_data->save_path + "/GT.ot");
		//GT_sample_voxels
		share_data->GT_sample->updateInnerOccupancy();
		//share_data->GT_sample->write(share_data->save_path + "/GT_sample.ot");
		share_data->init_voxels = 0;
		for (octomap::ColorOcTree::leaf_iterator it = share_data->GT_sample->begin_leafs(), end = share_data->GT_sample->end_leafs(); it != end; ++it) {
			share_data->init_voxels++;
		}
		//cout << "Map_GT_sample has voxels " << share_data->init_voxels << endl;
		//if (share_data->init_voxels != 32768) cout << "WARNING! BBX small." << endl;
		//ofstream fout(share_data->save_path + "/GT_size.txt");
		//fout << scale * predicted_size << endl;
		share_data->full_voxels = 0;
		for (octomap::ColorOcTree::leaf_iterator it = share_data->ground_truth_model->begin_leafs(), end = share_data->ground_truth_model->end_leafs(); it != end; ++it) {
			share_data->full_voxels++;
		}
		//初始化viewspace
		view_space = make_shared<View_Space>(share_data);
		//show mesh space
		if (share_data->show) {
			pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("mesh"));
			viewer1->setBackgroundColor(255, 255, 255);
			viewer1->addCoordinateSystem(0.1);
			viewer1->initCameraParameters();
			double max_z = 0;
			int index_z = 0;
			for (int i = 0; i < view_space->views.size(); i++) {
				Eigen::Vector4d X(0.01, 0, 0, 1);
				Eigen::Vector4d Y(0, 0.01, 0, 1);
				Eigen::Vector4d Z(0, 0, 0.01, 1);
				Eigen::Vector4d O(0, 0, 0, 1);
				view_space->views[i].get_next_camera_pos(Eigen::Matrix4d::Identity(4, 4), object_center_world);
				Eigen::Matrix4d view_pose_world = (Eigen::Matrix4d::Identity(4, 4) * view_space->views[i].pose.inverse()).eval();
				//cout << view_pose_world << endl;
				X = view_pose_world * X;
				Y = view_pose_world * Y;
				Z = view_pose_world * Z;
				O = view_pose_world * O;
				viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i));
				viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i));
				viewer1->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i));
				viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "X" + to_string(i));
				viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Y" + to_string(i));
				viewer1->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "Z" + to_string(i));

				if (view_space->views[i].init_pos(1) > max_z) {
					max_z = view_space->views[i].init_pos(2);
					index_z = i;
				}
			}
			cout << "z_max_index is " << index_z << endl;
			viewer1->addPolygonMesh(*share_data->mesh_ply, "mesh_ply");
			view_space->add_bbx_to_cloud(viewer1);
			while (!viewer1->wasStopped())
			{
				viewer1->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds(100000));
			}
			viewer1->removeAllCoordinateSystems();
			viewer1->removeAllPointClouds();
			viewer1->removeAllShapes();
			viewer1->close();
			viewer1.reset();
		}
		//相机类初始化
		percept = make_shared<Perception_3D>(share_data);
	}

	void get_cover_view_cases() {
		double now_time = clock();
		//从球面生成半球面coverage视点空间
		for (int i = 6; i <= 1080; i++) {
		//for (int i = 750; i <= 750; i++) {
			vector<Eigen::Vector4d> view_points_uniform;
			ifstream fin_sphere(share_data->orginalviews_path + to_string(i) + ".txt");
			if (!fin_sphere) {
				cout << "skip. cannot open file " << share_data->orginalviews_path + to_string(i) + ".txt" << endl;
				continue;
			}
			int num,id; 
			double x, y, z, dis, angel;
			fin_sphere >> num >> dis >> angel;
			for (int j = 0; j < i; j++) {
				fin_sphere >> id >> x >> y >> z;
				double r = sqrt(x * x + y * y + z * z);
				view_points_uniform.push_back(Eigen::Vector4d(x / r, y / r, z / r, 1.0));
			}
			for (int k = 0; k < i; k++) {
				// 旋转视点空间，使得第k个视点旋转为坐标 (0, sqrt(3)/2, 1/2)，保持其他视点依旧在单位球上
				Eigen::Vector3d target_point(0, std::sqrt(3.0) / 2.0, 1.0 / 2.0);
				Eigen::Vector3d view_point = Eigen::Vector3d(view_points_uniform[k](0), view_points_uniform[k](1), view_points_uniform[k](2));
				// 计算旋转轴和角度
				Eigen::Vector3d axis = view_point.cross(target_point).normalized();
				double angle = std::acos(view_point.normalized().dot(target_point.normalized()));
				// 使用Quaternion计算旋转
				Eigen::Quaterniond quaternion;
				quaternion = Eigen::AngleAxisd(angle, axis);
				// 将Quaternion转换为旋转矩阵
				Eigen::Matrix3d rotation_matrix = quaternion.toRotationMatrix();
				Eigen::Matrix4d R = Eigen::Matrix4d::Identity(4, 4);
				R.block(0, 0, 3, 3) = rotation_matrix;
				//旋转视点空间
				vector<Eigen::Vector3d> out_view_points;
				for (int j = 0; j < i; j++) {
					view_points_uniform[j] = R * view_points_uniform[j];
					if (view_points_uniform[j](2) >= 0) out_view_points.push_back(Eigen::Vector3d(view_points_uniform[j](0), view_points_uniform[j](1), view_points_uniform[j](2)));
				}
				//保存视点空间
				ifstream fin_vs(share_data->viewspace_path + to_string(out_view_points.size()) + ".txt");
				if (!fin_vs.is_open()) {
					fin_vs.close();
					ofstream fout_vs(share_data->viewspace_path + to_string(out_view_points.size()) + ".txt");
					for (int j = 0; j < out_view_points.size(); j++) {
						fout_vs << out_view_points[j](0) << ' ' << out_view_points[j](1) << ' ' << out_view_points[j](2) << '\n';
					}
					fout_vs.close();
				}
				else {
					vector<Eigen::Vector3d> pre_view_points;
					double x, y, z;
					while (fin_vs >> x >> y >> z) {
						pre_view_points.push_back(Eigen::Vector3d(x, y, z));
					}
					fin_vs.close();
					double pre_dis = 0, dis = 0;
					for (int x = 0; x < out_view_points.size(); x++)
						for (int y = x + 1; y < out_view_points.size(); y++) {
							pre_dis += (pre_view_points[x] - pre_view_points[y]).norm();
							dis += (out_view_points[x] - out_view_points[y]).norm();
						}
					if (dis >= pre_dis) {
						ofstream fout_vs(share_data->viewspace_path + to_string(out_view_points.size()) + ".txt");
						for (int j = 0; j < out_view_points.size(); j++) {
							fout_vs << out_view_points[j](0) << ' ' << out_view_points[j](1) << ' ' << out_view_points[j](2) << '\n';
						}
						fout_vs.close();
					}
				}
			}
		}

		//对于2,3,4,5单独输出
		ofstream fout_vs_2(share_data->viewspace_path + to_string(2) + ".txt");
		fout_vs_2 << "0 0.86602540378 0.5\n";
		fout_vs_2 << "0 -0.86602540378 0.5\n";
		fout_vs_2.close();
		ofstream fout_vs_3(share_data->viewspace_path + to_string(3) + ".txt");
		fout_vs_3 << "0 0.86602540378 0.5\n";
		fout_vs_3 << "0 -0.86602540378 0.5\n";
		fout_vs_3 << "0.86602540378 0 0.5\n";
		fout_vs_3.close();
		ofstream fout_vs_4(share_data->viewspace_path + to_string(4) + ".txt");
		fout_vs_4 << "0 0.86602540378 0.5\n";
		fout_vs_4 << "0 -0.86602540378 0.5\n";
		fout_vs_4 << "0.86602540378 0 0.5\n";
		fout_vs_4 << "-0.86602540378 0 0.5\n";
		fout_vs_4.close();
		ofstream fout_vs_5(share_data->viewspace_path + to_string(5) + ".txt");
		fout_vs_5 << "0 0.86602540378 0.5\n";
		fout_vs_5 << "0 -0.86602540378 0.5\n";
		fout_vs_5 << "0.86602540378 0 0.5\n";
		fout_vs_5 << "-0.86602540378 0 0.5\n";
		fout_vs_5 << "0 0 1\n";

		cout << "view cases get with executed time " << clock() - now_time << " ms." << endl;
	}

	int get_coverage() {
		double now_time = clock();
		//json root
		Json::Value root;
		root["camera_angle_x"] = 2.0 * atan(0.5 * share_data->color_intrinsics.width / share_data->color_intrinsics.fx);
		root["camera_angle_y"] = 2.0 * atan(0.5 * share_data->color_intrinsics.height / share_data->color_intrinsics.fy);
		root["fl_x"] = share_data->color_intrinsics.fx;
		root["fl_y"] = share_data->color_intrinsics.fy;
		root["k1"] = share_data->color_intrinsics.coeffs[0];
		root["k2"] = share_data->color_intrinsics.coeffs[1];
		root["k3"] = share_data->color_intrinsics.coeffs[2];
		root["p1"] = share_data->color_intrinsics.coeffs[3];
		root["p2"] = share_data->color_intrinsics.coeffs[4];
		root["cx"] = share_data->color_intrinsics.ppx;
		root["cy"] = share_data->color_intrinsics.ppy;
		root["w"] = share_data->color_intrinsics.width;
		root["h"] = share_data->color_intrinsics.height;
		root["aabb_scale"] = share_data->ray_casting_aabb_scale;
		root["scale"] = 0.5 / share_data->predicted_size;
		root["offset"][0] = 0.5 + share_data->object_center_world(2);
		root["offset"][1] = 0.5 + share_data->object_center_world(0);
		root["offset"][2] = 0.5 + share_data->object_center_world(1);
		
		//每个视点成像并写入文件
		for (int i = 0; i < share_data->num_of_views; i++) {
			//get point cloud
			//percept->precept(view_space->views[i]);
			share_data->access_directory(share_data->gt_path + "/" + to_string(share_data->num_of_views));
			//get rgb image
			percept->render(view_space->views[i], i, "/" + to_string(share_data->num_of_views));
			cv::Mat rgb_image = cv::imread(share_data->gt_path + "/" + to_string(share_data->num_of_views) + "/rgb_" + to_string(i) + ".png", cv::IMREAD_UNCHANGED);
			cv::Mat rgb_image_alpha, rgb_image_alpha_clip;
			convertToAlpha(rgb_image, rgb_image_alpha);
			//cv::imwrite(share_data->gt_path + "/" + to_string(share_data->num_of_views) + "/rgba_" + to_string(i) + ".png", rgb_image_alpha);
			rgb_image_alpha_clip = rgb_image_alpha.clone();
			cv::flip(rgb_image_alpha_clip, rgb_image_alpha_clip, -1);
			cv::imwrite(share_data->gt_path + "/" + to_string(share_data->num_of_views) + "/rgbaClip_" + to_string(i) + ".png", rgb_image_alpha_clip);
			remove((share_data->gt_path + "/" + to_string(share_data->num_of_views) + "/rgb_" + to_string(i) + ".png").c_str());
			//get json
			Json::Value view_image;
			view_image["file_path"] = to_string(share_data->num_of_views) + "/rgbaClip_" + to_string(i) + ".png";
			Json::Value transform_matrix;
			for (int k = 0; k < 4; k++) {
				Json::Value row;
				for (int l = 0; l < 4; l++) {
					view_space->views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
					Eigen::Matrix4d view_pose_world = share_data->now_camera_pose_world * view_space->views[i].pose.inverse();
					//x,y,z->y,z,x
					Eigen::Matrix4d pose;
					pose << 0, 0, 1, 0,
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 0, 1;
					//x,y,z->x,-y,-z
					Eigen::Matrix4d pose_1;
					pose_1 << 1, 0, 0, 0,
						0, -1, 0, 0,
						0, 0, -1, 0,
						0, 0, 0, 1;
					view_pose_world = pose * view_pose_world * pose_1;
					row.append(view_pose_world(k, l));
				}
				transform_matrix.append(row);
			}
			view_image["transform_matrix"] = transform_matrix;
			root["frames"].append(view_image);
		}
		Json::StyledWriter writer;
		ofstream fout(share_data->gt_path + "/" + to_string(share_data->num_of_views) + ".json");
		fout << writer.write(root);
		fout.close();

		cout << "images get with executed time " << clock() - now_time << " ms." << endl;

		return 0;
	}

	int train_by_instantNGP(string trian_json_file, string test_json_file = "100", bool nbv_test = false, int ensemble_id = -1) {
		double now_time = clock();
		//使用命令行训练
		ofstream fout_py(share_data->instant_ngp_path + "interact/run_with_c++.py");

		fout_py << "import os" << endl;

		string cmd = "python " + share_data->instant_ngp_path + "run.py";
		//cmd += " --gui";
		cmd += " --train";
		cmd += " --n_steps " + to_string(share_data->n_steps);

		if (!nbv_test) {
			cmd += " --scene " + share_data->gt_path + "/" + trian_json_file + ".json ";
			cmd += " --test_transforms " + share_data->gt_path + "/" + test_json_file + ".json ";
			cmd += " --save_metrics " + share_data->gt_path + "/" + trian_json_file + ".txt ";
		}
		else {
			cmd += " --scene " + share_data->save_path + "/json/" + trian_json_file + ".json ";
			if (ensemble_id == -1) {
				cmd += " --test_transforms " + share_data->gt_path + "/" + test_json_file + ".json ";
				cmd += " --save_metrics " + share_data->save_path + "/metrics/" + trian_json_file + ".txt ";
			}
			else {
				cmd += " --screenshot_transforms " + share_data->save_path + "/render_json/" + trian_json_file + ".json ";
				cmd += " --screenshot_dir " + share_data->save_path + "/render/" + trian_json_file + "/ensemble_" + to_string(ensemble_id) + "/";
			}
		}

		string python_cmd = "os.system(\'" + cmd + "\')";
		fout_py << python_cmd << endl;
		fout_py.close();

		ofstream fout_py_ready(share_data->instant_ngp_path + "interact/ready_c++.txt");
		fout_py_ready.close();

		ifstream fin_over(share_data->instant_ngp_path + "interact/ready_py.txt");
		while (!fin_over.is_open()) {
			boost::this_thread::sleep(boost::posix_time::seconds(1));
			fin_over.open(share_data->instant_ngp_path + "interact/ready_py.txt");
		}
		fin_over.close();
		boost::this_thread::sleep(boost::posix_time::seconds(1));
		remove((share_data->instant_ngp_path + "interact/ready_py.txt").c_str());

		double cost_time = (clock() - now_time) / CLOCKS_PER_SEC;
		cout << "train and eval with executed time " << cost_time << " s." << endl;

		if (nbv_test) {
			if (ensemble_id == -1) {
				ofstream fout_time(share_data->save_path + "/train_time/" + trian_json_file + ".txt");
				fout_time << cost_time << endl;
				fout_time.close();
			}
		}

		return 0;
	}

	int sample_colored_clouds(string view_idx = "0", string generation_ensemble_idx = "0") {
		double now_time = clock();

		string obj_file_name = view_idx + "/" + generation_ensemble_idx;

		string flip_code = view_idx;

		//使用命令行调用
		share_data->access_directory(share_data->one12345pp_path + "interact/");

		ofstream fout_py(share_data->one12345pp_path + "interact/run_with_c++.py");

		fout_py << "import os" << endl;

		string cmd;
		string python_cmd;

		//obj2pcd
		cmd = "python " + share_data->one12345pp_path + "obj2pcd.py " + share_data->name_of_pcd + "/" + obj_file_name + ".obj " + to_string(share_data->num_of_bbx_egde) + " " + flip_code;
		python_cmd = "os.system(\'" + cmd + "\')";
		fout_py << python_cmd << endl;

		fout_py.close();

		ofstream fout_py_ready(share_data->one12345pp_path + "interact/ready_c++.txt");
		fout_py_ready.close();

		ifstream fin_over(share_data->one12345pp_path + "interact/ready_py.txt");
		while (!fin_over.is_open()) {
			boost::this_thread::sleep(boost::posix_time::seconds(1));
			fin_over.open(share_data->one12345pp_path + "interact/ready_py.txt");
		}
		fin_over.close();
		boost::this_thread::sleep(boost::posix_time::seconds(1));
		remove((share_data->one12345pp_path + "interact/ready_py.txt").c_str());

		double cost_time = (clock() - now_time) / CLOCKS_PER_SEC;
		cout << "obj sampled with executed time " << cost_time << " s." << endl;

		ofstream fout_time(share_data->save_path + "/obj_time/" + obj_file_name + ".txt");
		fout_time << cost_time << endl;
		fout_time.close();

		return 0;
	}

	//first_view_id是测试视点空间（144）中的id，init_view_id是5覆盖情况下的初始视点集合
	int nbv_loop(int first_view_id = -1, int init_view_id = 0, int test_id = 0, bool eval_mode = false) {

		if (init_views.size() == 0) {
			cout << "init_views is empty. read init (5) coverage view space first." << endl;
			return -1;
		}

		if (first_view_id == -1) {
			first_view_id = 0;
			cout << "first_view_id is -1. use 0 as id." << endl;
		}

		if (share_data->method_of_IG != Imageto3DCovering) {
			cout << "method_of_IG is not Imageto3DCovering. Read view budget." << endl;
			string reference_method_string = "_m5_sc6_su0.10";
			if (share_data->do_ablation) {
				reference_method_string = "_m5_sc" + to_string(share_data->num_of_min_cover) + "_su0.10";
			}
			ifstream fin_view_budget(share_data->pre_path + "Compare/" + share_data->name_of_pcd + reference_method_string + "_v" + to_string(init_view_id) + "_t" + to_string(0) + "/view_budget.txt");
			if (fin_view_budget.is_open()) {
				int view_budget;
				fin_view_budget >> view_budget;
				fin_view_budget.close();
				share_data->num_of_max_iteration = view_budget - 1;
				cout << "readed view_budget is " << view_budget << endl;
			}
			else {
				cout << "view_budget.txt is not exist. use deaulft as view budget." << endl;
			}
			cout << "num_of_max_iteration is set as " << share_data->num_of_max_iteration << endl;
		}

		cout << "first_view_id is " << first_view_id << endl;
		cout << "init_view_id is " << init_view_id << endl;
		cout << "test_id is " << test_id << endl;
		cout << "num_of_max_iteration is " << share_data->num_of_max_iteration << endl;

		share_data->save_path += "_v" + to_string(init_view_id);
		share_data->save_path += "_t" + to_string(test_id);
		share_data->access_directory(share_data->save_path + "/json");
		share_data->access_directory(share_data->save_path + "/render_json");
		share_data->access_directory(share_data->save_path + "/metrics");
		share_data->access_directory(share_data->save_path + "/render");
		share_data->access_directory(share_data->save_path + "/train_time");
		share_data->access_directory(share_data->save_path + "/obj_time");
		share_data->access_directory(share_data->save_path + "/infer_time");
		share_data->access_directory(share_data->save_path + "/movement");
		share_data->access_directory(share_data->save_path + "/evaluation");
		share_data->access_directory(share_data->save_path + "/optimazation_time");

		if (!share_data->nbv_resume_mode) {
			ifstream check_fininshed(share_data->save_path + "/run_time.txt");
			if (check_fininshed.is_open()) {
				double run_time;
				check_fininshed >> run_time;
				check_fininshed.close();
				if (run_time >= 0) {
					if (eval_mode) {
						ifstream fin_metrics(share_data->save_path + "/metrics/" + to_string(share_data->num_of_max_iteration) + ".txt");
						if (!fin_metrics.is_open()) {
							cout << "final evaluating..." << endl;
							train_by_instantNGP(to_string(share_data->num_of_max_iteration), "100", true);
							string metric_name;
							double metric_value;
							while (fin_metrics >> metric_name >> metric_value) {
								cout << metric_name << ": " << metric_value << endl;
							}
							fin_metrics.close();
						}
						else {
							cout << "final metrics is exist. nbv_loop is finished." << endl;
						}
						return 0;
					}
					else {
						cout << "run_time.txt is exist. nbv_loop is finished." << endl;
						return 0;
					}
				}
			}
		}

		//json root
		Json::Value root_nbvs;
		root_nbvs["camera_angle_x"] = 2.0 * atan(0.5 * share_data->color_intrinsics.width / share_data->color_intrinsics.fx);
		root_nbvs["camera_angle_y"] = 2.0 * atan(0.5 * share_data->color_intrinsics.height / share_data->color_intrinsics.fy);
		root_nbvs["fl_x"] = share_data->color_intrinsics.fx;
		root_nbvs["fl_y"] = share_data->color_intrinsics.fy;
		root_nbvs["k1"] = share_data->color_intrinsics.coeffs[0];
		root_nbvs["k2"] = share_data->color_intrinsics.coeffs[1];
		root_nbvs["k3"] = share_data->color_intrinsics.coeffs[2];
		root_nbvs["p1"] = share_data->color_intrinsics.coeffs[3];
		root_nbvs["p2"] = share_data->color_intrinsics.coeffs[4];
		root_nbvs["cx"] = share_data->color_intrinsics.ppx;
		root_nbvs["cy"] = share_data->color_intrinsics.ppy;
		root_nbvs["w"] = share_data->color_intrinsics.width;
		root_nbvs["h"] = share_data->color_intrinsics.height;
		root_nbvs["aabb_scale"] = share_data->ray_casting_aabb_scale;
		root_nbvs["scale"] = 0.5 / share_data->predicted_size;
		root_nbvs["offset"][0] = 0.5 + share_data->object_center_world(2);
		root_nbvs["offset"][1] = 0.5 + share_data->object_center_world(0);
		root_nbvs["offset"][2] = 0.5 + share_data->object_center_world(1);

		//double div_rate_render = 16.0;
		double div_rate_render = share_data->div_rate_render;
		Json::Value root_render;
		root_render["camera_angle_x"] = 2.0 * atan(0.5 * share_data->color_intrinsics.width / share_data->color_intrinsics.fx);
		root_render["camera_angle_y"] = 2.0 * atan(0.5 * share_data->color_intrinsics.height / share_data->color_intrinsics.fy);
		root_render["fl_x"] = share_data->color_intrinsics.fx / div_rate_render;
		root_render["fl_y"] = share_data->color_intrinsics.fy / div_rate_render;
		root_render["k1"] = share_data->color_intrinsics.coeffs[0];
		root_render["k2"] = share_data->color_intrinsics.coeffs[1];
		root_render["k3"] = share_data->color_intrinsics.coeffs[2];
		root_render["p1"] = share_data->color_intrinsics.coeffs[3];
		root_render["p2"] = share_data->color_intrinsics.coeffs[4];
		root_render["cx"] = share_data->color_intrinsics.ppx / div_rate_render;
		root_render["cy"] = share_data->color_intrinsics.ppy / div_rate_render;
		root_render["w"] = share_data->color_intrinsics.width / div_rate_render;
		root_render["h"] = share_data->color_intrinsics.height / div_rate_render;
		root_render["aabb_scale"] = share_data->ray_casting_aabb_scale;
		root_render["scale"] = 0.5 / share_data->predicted_size;
		root_render["offset"][0] = 0.5 + share_data->object_center_world(2);
		root_render["offset"][1] = 0.5 + share_data->object_center_world(0);
		root_render["offset"][2] = 0.5 + share_data->object_center_world(1);

		if (share_data->method_of_IG == PVBCoverage) {
			Json::Value view_image;
			view_image["file_path"] = "../../../Coverage_images/" + share_data->name_of_pcd + "/" + to_string(share_data->num_of_views) + "/rgbaClip_" + to_string(first_view_id) + ".png";
			Json::Value transform_matrix;
			for (int k = 0; k < 4; k++) {
				Json::Value row;
				for (int l = 0; l < 4; l++) {
					view_space->views[first_view_id].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
					Eigen::Matrix4d view_pose_world = share_data->now_camera_pose_world * view_space->views[first_view_id].pose.inverse();
					//x,y,z->y,z,x
					Eigen::Matrix4d pose;
					pose << 0, 0, 1, 0,
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 0, 1;
					//x,y,z->x,-y,-z
					Eigen::Matrix4d pose_1;
					pose_1 << 1, 0, 0, 0,
						0, -1, 0, 0,
						0, 0, -1, 0,
						0, 0, 0, 1;
					view_pose_world = pose * view_pose_world * pose_1;
					row.append(view_pose_world(k, l));
				}
				transform_matrix.append(row);
			}
			view_image["transform_matrix"] = transform_matrix;
			root_nbvs["frames"].append(view_image);
		}

		share_data->render_intrinsics.width = share_data->color_intrinsics.width / div_rate_render;
		share_data->render_intrinsics.height = share_data->color_intrinsics.height / div_rate_render;
		share_data->render_intrinsics.fx = share_data->color_intrinsics.fx / div_rate_render;
		share_data->render_intrinsics.fy = share_data->color_intrinsics.fy / div_rate_render;
		share_data->render_intrinsics.ppx = share_data->color_intrinsics.ppx / div_rate_render;
		share_data->render_intrinsics.ppy = share_data->color_intrinsics.ppy / div_rate_render;
		share_data->render_intrinsics.model = share_data->color_intrinsics.model;
		share_data->render_intrinsics.coeffs[0] = share_data->color_intrinsics.coeffs[0];
		share_data->render_intrinsics.coeffs[1] = share_data->color_intrinsics.coeffs[1];
		share_data->render_intrinsics.coeffs[2] = share_data->color_intrinsics.coeffs[2];
		share_data->render_intrinsics.coeffs[3] = share_data->color_intrinsics.coeffs[3];
		share_data->render_intrinsics.coeffs[4] = share_data->color_intrinsics.coeffs[4];
		cout << "render_intrinsics: " << share_data->render_intrinsics.fx << " " << share_data->render_intrinsics.fy << " " << share_data->render_intrinsics.ppx << " " << share_data->render_intrinsics.ppy << endl;

		double total_movement_cost = 0.0;
		ofstream fout_move_first(share_data->save_path + "/movement/" + to_string(-1) + ".txt");
		fout_move_first << first_view_id << '\t' << 0 << '\t' << total_movement_cost << endl;
		fout_move_first.close();

		//初始视点
		vector<int> chosen_nbvs;
		chosen_nbvs.push_back(first_view_id);
		set<int> chosen_nbvs_set;
		chosen_nbvs_set.insert(first_view_id);

		vector<int> oneshot_views;

		//循环NBV
		double now_time = clock();
		int iteration = 0;
		while (true) {
			//生成当前视点集合json
			Json::Value now_nbvs_json(root_nbvs);
			Json::Value now_render_json(root_render);
			for (int i = 0; i < share_data->num_of_views; i++) {
				Json::Value view_image;
				view_image["file_path"] = "../../../Coverage_images/" + share_data->name_of_pcd + "/" + to_string(share_data->num_of_views) + "/rgbaClip_" + to_string(i) + ".png";
				Json::Value transform_matrix;
				for (int k = 0; k < 4; k++) {
					Json::Value row;
					for (int l = 0; l < 4; l++) {
						view_space->views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
						Eigen::Matrix4d view_pose_world = share_data->now_camera_pose_world * view_space->views[i].pose.inverse();
						//x,y,z->y,z,x
						Eigen::Matrix4d pose;
						pose << 0, 0, 1, 0,
							1, 0, 0, 0,
							0, 1, 0, 0,
							0, 0, 0, 1;
						//x,y,z->x,-y,-z
						Eigen::Matrix4d pose_1;
						pose_1 << 1, 0, 0, 0,
							0, -1, 0, 0,
							0, 0, -1, 0,
							0, 0, 0, 1;
						view_pose_world = pose * view_pose_world * pose_1;
						row.append(view_pose_world(k, l));
					}
					transform_matrix.append(row);
				}
				view_image["transform_matrix"] = transform_matrix;
				if (chosen_nbvs_set.count(i)) now_nbvs_json["frames"].append(view_image);
				else now_render_json["frames"].append(view_image);
			}
			Json::StyledWriter writer_nbvs_json;
			ofstream fout_nbvs_json(share_data->save_path + "/json/" + to_string(iteration) + ".json");
			fout_nbvs_json << writer_nbvs_json.write(now_nbvs_json);
			fout_nbvs_json.close();
			Json::StyledWriter writer_render_json;
			ofstream fout_render_json(share_data->save_path + "/render_json/" + to_string(iteration) + ".json");
			fout_render_json << writer_render_json.write(now_render_json);
			fout_render_json.close();

			//如果需要测试，则训练当前视点集合
			cout << "iteration " << iteration << endl;
			cout<< "chosen_nbvs: ";
			for (int i = 0; i < chosen_nbvs.size(); i++) {
				cout << chosen_nbvs[i] << ' ';
			}
			cout << endl;
			if (share_data->evaluate) {
				cout << "evaluating..." << endl;
				train_by_instantNGP(to_string(iteration), "100", true);
				ifstream fin_metrics(share_data->save_path + "/metrics/" + to_string(iteration) + ".txt");
				string metric_name;
				double metric_value;
				while (fin_metrics >> metric_name >> metric_value) {
					cout << metric_name << ": " << metric_value << endl;
				}
				fin_metrics.close();
			}

			//如果达到最大迭代次数，则结束
			if (iteration == share_data->num_of_max_iteration) {
				//保存运行时间
				double loops_time = (clock() - now_time) / CLOCKS_PER_SEC;
				ofstream fout_loops_time(share_data->save_path + "/run_time.txt");
				fout_loops_time << loops_time << endl;
				fout_loops_time.close();
				cout << "run time " << loops_time << " ms." << endl;
				//如果不需要逐步测试，则训练最终视点集合
				if (!share_data->evaluate) {
					cout << "final evaluating..." << endl;
					train_by_instantNGP(to_string(iteration), "100", true);
					ifstream fin_metrics(share_data->save_path + "/metrics/" + to_string(iteration) + ".txt");
					string metric_name;
					double metric_value;
					while (fin_metrics >> metric_name >> metric_value) {
						cout << metric_name << ": " << metric_value << endl;
					}
					fin_metrics.close();
				}
				break;
			}

			//根据不同方法获取NBV
			double infer_time = clock();
			int next_view_id = -1;
			double largest_view_uncertainty = -1e100;
			int best_view_id = -1;
			bool nbv_resumed = false;
			switch (share_data->method_of_IG) {
				case 0: //RandomIterative
					next_view_id = rand() % share_data->num_of_views;
					while (chosen_nbvs_set.count(next_view_id)) {
						next_view_id = rand() % share_data->num_of_views;
					}
					break;

				case 1: //RandomOneshot
					if (oneshot_views.size() == 0) {
						set<int> oneshot_views_set;
						oneshot_views_set.insert(first_view_id);
						for (int i = 0; i < share_data->num_of_max_iteration; i++) {
							int random_view_id = rand() % share_data->num_of_views;
							while (oneshot_views_set.count(random_view_id)) {
								random_view_id = rand() % share_data->num_of_views;
							}
							oneshot_views_set.insert(random_view_id);
						}
						for (auto it = oneshot_views_set.begin(); it != oneshot_views_set.end(); it++) {
							oneshot_views.push_back(*it);
						}
						cout << "oneshot_views num is: " << oneshot_views.size() << endl;
						Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data, view_space->views, oneshot_views, first_view_id);
						double total_dis = gloabl_path_planner->solve();
						oneshot_views = gloabl_path_planner->get_path_id_set();
						if (oneshot_views.size() != share_data->num_of_max_iteration + 1) {
							cout << "oneshot_views.size() != share_data->num_of_max_iteration + 1" << endl;
						}
						cout<< "total_dis: " << total_dis << endl;
						delete gloabl_path_planner;
						//删除初始视点
						oneshot_views.erase(oneshot_views.begin());
						//更新迭代次数，取出NBV
						share_data->num_of_max_iteration = oneshot_views.size();
						next_view_id = oneshot_views[0];
						oneshot_views.erase(oneshot_views.begin());
					}
					else {
						next_view_id = oneshot_views[0];
						oneshot_views.erase(oneshot_views.begin());
					}
					break;

				case 2: //EnsembleRGB
					nbv_resumed = false;
					if (share_data->nbv_resume_mode) {
						//检查当前迭代是否存在
						ifstream fin_nbv(share_data->save_path + "/movement/" + to_string(iteration) + ".txt");
						if (fin_nbv) {
							fin_nbv >> best_view_id;
							fin_nbv.close();
						}
						if (best_view_id >=0 && best_view_id < share_data->num_of_views) {
							nbv_resumed = true;
							cout << "nbv_resumed: " << best_view_id << endl;
						}
					}
					if (!nbv_resumed) {
						//交给instantngp训练
						for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
							train_by_instantNGP(to_string(iteration), "100", true, ensemble_id);
						}
						//计算评价指标
						for (int i = 0; i < share_data->num_of_views; i++) {
							if (chosen_nbvs_set.count(i)) continue;
							vector<cv::Mat> rgb_images;
							for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
								cv::Mat rgb_image = cv::imread(share_data->save_path + "/render/" + to_string(iteration) + "/ensemble_" + to_string(ensemble_id) + "/rgbaClip_" + to_string(i) + ".png", cv::IMREAD_UNCHANGED);
								rgb_images.push_back(rgb_image);
							}
							//使用ensemble计算uncertainty
							double view_uncertainty = 0.0;
							for (int j = 0; j < rgb_images[0].rows; j++) {
								for (int k = 0; k < rgb_images[0].cols; k++) {
									double mean_r = 0.0;
									double mean_g = 0.0;
									double mean_b = 0.0;
									for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
										cv::Vec4b rgba = rgb_images[ensemble_id].at<cv::Vec4b>(j, k);
										//注意不要归一化，不然会导致取log为负
										mean_r += rgba[0];
										mean_g += rgba[1];
										mean_b += rgba[2];
									}
									//计算方差
									mean_r /= share_data->ensemble_num;
									mean_g /= share_data->ensemble_num;
									mean_b /= share_data->ensemble_num;
									double variance_r = 0.0;
									double variance_g = 0.0;
									double variance_b = 0.0;
									for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
										cv::Vec4b rgba = rgb_images[ensemble_id].at<cv::Vec4b>(j, k);
										variance_r += (rgba[0] - mean_r) * (rgba[0] - mean_r);
										variance_g += (rgba[1] - mean_g) * (rgba[1] - mean_g);
										variance_b += (rgba[2] - mean_b) * (rgba[2] - mean_b);
									};
									variance_r /= share_data->ensemble_num;
									variance_g /= share_data->ensemble_num;
									variance_b /= share_data->ensemble_num;
									if (variance_r > 1e-10) view_uncertainty += log(variance_r);
									if (variance_g > 1e-10) view_uncertainty += log(variance_g);
									if (variance_b > 1e-10) view_uncertainty += log(variance_b);
								}
							}
							//cout << i << " " << view_uncertainty << " " << largest_view_uncertainty << endl;
							if (view_uncertainty > largest_view_uncertainty) {
								largest_view_uncertainty = view_uncertainty;
								best_view_id = i;
							}
							rgb_images.clear();
							rgb_images.shrink_to_fit();
						}
					}
					//选择最好的视点
					next_view_id = best_view_id;
					break;
				
				case 3: //EnsembleRGBDensity	
					nbv_resumed = false;
					if (share_data->nbv_resume_mode) {
						//检查当前迭代是否存在
						ifstream fin_nbv(share_data->save_path + "/movement/" + to_string(iteration) + ".txt");
						if (fin_nbv) {
							fin_nbv >> best_view_id;
							fin_nbv.close();
						}
						if (best_view_id >= 0 && best_view_id < share_data->num_of_views) {
							nbv_resumed = true;
							cout << "nbv_resumed: " << best_view_id << endl;
						}
					}
					if (!nbv_resumed) {
						//交给instantngp训练
						for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
							train_by_instantNGP(to_string(iteration), "100", true, ensemble_id);
						}
						//计算评价指标
						for (int i = 0; i < share_data->num_of_views; i++) {
							if (chosen_nbvs_set.count(i)) continue;
							vector<cv::Mat> rgb_images;
							for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
								cv::Mat rgb_image = cv::imread(share_data->save_path + "/render/" + to_string(iteration) + "/ensemble_" + to_string(ensemble_id) + "/rgbaClip_" + to_string(i) + ".png", cv::IMREAD_UNCHANGED);
								rgb_images.push_back(rgb_image);
							}
							//使用ensemble计算uncertainty，其中density存于alpha通道
							double view_uncertainty = 0.0;
							for (int j = 0; j < rgb_images[0].rows; j++) {
								for (int k = 0; k < rgb_images[0].cols; k++) {
									double mean_r = 0.0;
									double mean_g = 0.0;
									double mean_b = 0.0;
									double mean_density = 0.0;
									for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
										cv::Vec4b rgba = rgb_images[ensemble_id].at<cv::Vec4b>(j, k);
										//注意不要归一化，不然会导致取log为负
										mean_r += rgba[0];
										mean_g += rgba[1];
										mean_b += rgba[2];
										//注意alpha通道存储的是density，要归一化到0-1
										mean_density += rgba[3] / 255.0;
									}
									mean_r /= share_data->ensemble_num;
									mean_g /= share_data->ensemble_num;
									mean_b /= share_data->ensemble_num;
									mean_density /= share_data->ensemble_num;
									//cout << mean_r << " " << mean_g << " " << mean_b << " " << mean_density << endl;
									//计算方差
									double variance_r = 0.0;
									double variance_g = 0.0;
									double variance_b = 0.0;
									for (int ensemble_id = 0; ensemble_id < share_data->ensemble_num; ensemble_id++) {
										cv::Vec4b rgba = rgb_images[ensemble_id].at<cv::Vec4b>(j, k);
										variance_r += (rgba[0] - mean_r) * (rgba[0] - mean_r);
										variance_g += (rgba[1] - mean_g) * (rgba[1] - mean_g);
										variance_b += (rgba[2] - mean_b) * (rgba[2] - mean_b);
									};
									variance_r /= share_data->ensemble_num;
									variance_g /= share_data->ensemble_num;
									variance_b /= share_data->ensemble_num;
									view_uncertainty += (variance_r + variance_g + variance_b) / 3.0;
									view_uncertainty += (1.0 - mean_density) * (1.0 - mean_density);
								}
							}
							//cout << i << " " << view_uncertainty << " " << largest_view_uncertainty << endl;
							if (view_uncertainty > largest_view_uncertainty) {
								largest_view_uncertainty = view_uncertainty;
								best_view_id = i;
							}
							rgb_images.clear();
							rgb_images.shrink_to_fit();
						}
					}
					//选择最好的视点
					next_view_id = best_view_id;
					break;

				case 4: //PVBCoverage
					if (oneshot_views.size() == 0) {
						////通过网络获取视点预算
						share_data->access_directory(share_data->pvb_path + "data/images");
						ofstream fout_image(share_data->pvb_path + "data/images/" + to_string(0) + ".png", std::ios::binary);
						ifstream fin_image(share_data->gt_path + "/" + to_string(init_views.size()) + "/rgbaClip_" + to_string(init_view_id) + ".png", std::ios::binary);
						fout_image << fin_image.rdbuf();
						fout_image.close();
						fin_image.close();
						ofstream fout_ready(share_data->pvb_path + "data/ready_c++.txt");
						fout_ready.close();
						//等待python程序结束
						ifstream fin_over(share_data->pvb_path + "data/ready_py.txt");
						while (!fin_over.is_open()) {
							boost::this_thread::sleep(boost::posix_time::milliseconds(100));
							fin_over.open(share_data->pvb_path + "data/ready_py.txt");
						}
						fin_over.close();
						boost::this_thread::sleep(boost::posix_time::milliseconds(100));
						remove((share_data->pvb_path + "data/ready_py.txt").c_str());
						////读取view budget, bus25 gt20, airplane0 gt14
						int view_budget = -1;
						ifstream fin_view_budget(share_data->pvb_path + "data/view_budget.txt");
						if (!fin_view_budget.is_open()) {
							cout << "view budget file not found!" << endl;
						}
						fin_view_budget >> view_budget;
						fin_view_budget.close();
						cout << "view budget is: " << view_budget << endl;
						//读取当前视点
						View first_view = view_space->views[first_view_id];
						//读取coverage view space
						share_data->num_of_views = view_budget;
						ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
						share_data->pt_sphere.clear();
						share_data->pt_sphere.resize(share_data->num_of_views);
						for (int i = 0; i < share_data->num_of_views; i++) {
							share_data->pt_sphere[i].resize(3);
							for (int j = 0; j < 3; j++) {
								fin_sphere >> share_data->pt_sphere[i][j];
							}
						}
						cout << "coverage view space size is: " << share_data->pt_sphere.size() << endl;
						Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
						share_data->pt_norm = pt0.norm();
						view_space.reset();
						view_space = make_shared<View_Space>(share_data);
						int now_first_view_id = share_data->num_of_views;
						view_space->views.push_back(first_view);
						for (int i = 0; i < share_data->num_of_views + 1; i++) {
							oneshot_views.push_back(i);
						}
						chosen_nbvs.clear();
						chosen_nbvs.push_back(now_first_view_id);
						chosen_nbvs_set.clear();
						chosen_nbvs_set.insert(now_first_view_id);
						//执行全局路径规划
						Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data, view_space->views, oneshot_views, now_first_view_id);
						double total_dis = gloabl_path_planner->solve();
						oneshot_views = gloabl_path_planner->get_path_id_set();
						cout << "total_dis: " << total_dis << endl;
						delete gloabl_path_planner;
						//保存所有视点个数
						ofstream fout_iteration(share_data->save_path + "/view_budget.txt");
						fout_iteration << oneshot_views.size() << endl;
						//删除初始视点
						oneshot_views.erase(oneshot_views.begin());
						//更新迭代次数，取出NBV
						share_data->num_of_max_iteration = oneshot_views.size();
						next_view_id = oneshot_views[0];
						oneshot_views.erase(oneshot_views.begin());
					}
					else {
						next_view_id = oneshot_views[0];
						oneshot_views.erase(oneshot_views.begin());
					}
					break;

				case 5: //Imageto3DCovering, using 12345++ generated textured mesh as a prior
					if (oneshot_views.size() == 0) {
						//复制对应的图片，并进行对应旋转
						share_data->access_directory(share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/");
						ofstream fout_image(share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/.png", std::ios::binary);
						ifstream fin_image(share_data->gt_path + "/" + to_string(init_views.size()) + "/rgbaClip_" + to_string(init_view_id) + ".png", std::ios::binary);
						fout_image << fin_image.rdbuf();
						fout_image.close();
						fin_image.close();
						cv::Mat rgba_image = cv::imread(share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/.png", cv::IMREAD_UNCHANGED);
						cv::Mat rgb_image_alpha_rotated = rgba_image.clone();
						switch (init_view_id) {
						case 0: //不旋转
							break;
						case 1: //旋转180度
							cv::flip(rgba_image, rgb_image_alpha_rotated, -1);
							break;
						case 2: //逆时针旋转90度
							cv::transpose(rgba_image, rgb_image_alpha_rotated);
							cv::flip(rgb_image_alpha_rotated, rgb_image_alpha_rotated, 0);
							break;
						case 3: //顺时针旋转90度
							cv::transpose(rgba_image, rgb_image_alpha_rotated);
							cv::flip(rgb_image_alpha_rotated, rgb_image_alpha_rotated, 1);
							break;
						}
						cv::imwrite(share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/rotated.png", rgb_image_alpha_rotated);
						remove((share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/.png").c_str());
						//调用python脚本，infer 12345++，并获取渲染结果
						for (int i = 0; i < share_data->num_of_generation_ensemble; i++) {
							//inference 12345++，目前手动下载
							//do someting with sudoai api and convert to obj
							cout << "Please upload the image file " << share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/rotated.png" << endl;
							cout << "To https://www.sudo.ai/3dgen and download generated 3D mesh with .obj extension." << endl;
							cout << "Unzip and put all files to " << share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/" << endl;
							cout << "Rename the downloaded *.obj to " << share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/" + to_string(i) + ".obj" << endl;
							ifstream fin_obj(share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/" + to_string(i) + ".obj");
							while (!fin_obj.is_open()) {
								boost::this_thread::sleep(boost::posix_time::milliseconds(100));
								fin_obj.open(share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/" + to_string(i) + ".obj");
							}
							fin_obj.close();
							cout << "obj mesh readed." << endl;
							//调用python脚本，获取渲染结果
							sample_colored_clouds(to_string(init_view_id), to_string(i));
						}
						//读取采样的点云
						share_data->colored_pointclouds.resize(share_data->num_of_generation_ensemble);
						for (int i = 0; i < share_data->num_of_generation_ensemble; i++) {
							//读取pcd文件
							string pcd_path = share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/" + to_string(i) + ".pcd";
							pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
							if (share_data->use_gt_mesh) {
								*temp_cloud = *share_data->cloud_ground_truth;
							}
							else {
								if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcd_path, *temp_cloud) == -1) {
									cout << "Error: cannot read pcd file! " << pcd_path << endl;
									return -1;
								}
							}
							share_data->colored_pointclouds[i] = temp_cloud;
							if (share_data->show) {
								percept->viewer->addPointCloud<pcl::PointXYZRGB>(share_data->colored_pointclouds[0], "cloud_gt");
								percept->viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, share_data->points_size_cloud, "cloud_gt");
								//pcl::io::savePCDFile(share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/" + to_string(i) + "_scaled.pcd", *temp_cloud);
								while (!percept->viewer->wasStopped()) {
									percept->viewer->spinOnce(100);
									boost::this_thread::sleep(boost::posix_time::microseconds(100000));
								}
							}
						}
						double ray_start_time, ray_stop_time; 
						double opt_start_time, opt_stop_time;
						//只考虑使用scop求解，即只有几何覆盖
						ray_start_time = clock();
						views_voxels_set_covering* set_covering_optimizer = new views_voxels_set_covering(share_data.get(), view_space.get(), percept.get());
						ray_stop_time = clock();
						//利用优化器求解3D覆盖
						opt_start_time = clock();
						set_covering_optimizer->solve();
						opt_stop_time = clock();
						oneshot_views = set_covering_optimizer->get_view_id_set();
						cout << "oneshot_views num is: " << oneshot_views.size() << endl;
						delete set_covering_optimizer;
						//输出优化时间
						ofstream fout_ray_time(share_data->save_path + "/optimazation_time/ray.txt");
						fout_ray_time << (ray_stop_time - ray_start_time) / CLOCKS_PER_SEC << endl;
						ofstream fout_opt_time(share_data->save_path + "/optimazation_time/opt.txt");
						fout_opt_time << (opt_stop_time - opt_start_time) / CLOCKS_PER_SEC << endl;
						//释放优化数据
						delete share_data->voxel_id_map;

						//求解全局路径规划
						Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data, view_space->views, oneshot_views, first_view_id);
						double total_dis = gloabl_path_planner->solve();
						oneshot_views = gloabl_path_planner->get_path_id_set();
						cout << "total_dis: " << total_dis << endl;
						delete gloabl_path_planner;
						//保存所有视点个数
						ofstream fout_iteration(share_data->save_path + "/view_budget.txt");
						fout_iteration << oneshot_views.size() << endl;
						//删除初始视点
						oneshot_views.erase(oneshot_views.begin());
						//更新迭代次数，取出NBV
						share_data->num_of_max_iteration = oneshot_views.size();
						next_view_id = oneshot_views[0];
						oneshot_views.erase(oneshot_views.begin());
					}
					else {
						next_view_id = oneshot_views[0];
						oneshot_views.erase(oneshot_views.begin());
					}
					break;

			}
			chosen_nbvs.push_back(next_view_id);
			chosen_nbvs_set.insert(next_view_id);
			cout << "next_view_id: " << next_view_id << endl;

			if (!nbv_resumed) {
				infer_time = (clock() - infer_time) / CLOCKS_PER_SEC;
				ofstream fout_infer_time(share_data->save_path + "/infer_time/" + to_string(iteration) + ".txt");
				fout_infer_time << infer_time << endl;
				fout_infer_time.close();
			}

			//运动代价：视点id，当前代价，总体代价
			int now_nbv_id = chosen_nbvs[iteration];
			int next_nbv_id = chosen_nbvs[iteration + 1];
			pair<int, double> local_path = get_local_path(view_space->views[now_nbv_id].init_pos.eval(), view_space->views[next_nbv_id].init_pos.eval(), (share_data->object_center_world + Eigen::Vector3d(1e-10, 1e-10, 1e-10)).eval(), share_data->predicted_size);
			total_movement_cost += local_path.second;
			cout << "local path: " << local_path.second << " total: " << total_movement_cost << endl;

			ofstream fout_move(share_data->save_path + "/movement/" + to_string(iteration) + ".txt");
			fout_move << next_nbv_id << '\t' << local_path.second << '\t' << total_movement_cost << endl;
			fout_move.close();

			//更新迭代次数
			iteration++;
		}

		chosen_nbvs.clear();
		chosen_nbvs.shrink_to_fit();
		chosen_nbvs_set.clear();
		oneshot_views.clear();
		oneshot_views.shrink_to_fit();

		return 0;
	}

};

#define GetViewSpace 0
#define ViewPlanning 1
#define RenderDiffusion 2

int main()
{
	//Init
	srand(time(0));
	ios::sync_with_stdio(false);
	int mode;
	cout << "input mode:";
	cin >> mode;
	//测试集
	vector<string> names;
	cout << "input models:" << endl;
	string name;
	while (cin >> name) {
		if (name == "-1") break;
		names.push_back(name);
	}
	//选取模式
	if (mode == GetViewSpace) {
		shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", "");
		shared_ptr<View_Planning_Simulator> simulator = make_shared<View_Planning_Simulator>(share_data);
		simulator->get_cover_view_cases();
	}
	else if (mode == ViewPlanning) {
		vector<int> method_ids;
		//method_ids.push_back(0);
		//method_ids.push_back(4);
		method_ids.push_back(5);
		//method_ids.push_back(2);
		//method_ids.push_back(3);
		//method_ids.push_back(1);
		
		vector<int> init_view_ids;
		//init_view_ids.push_back(0);
		init_view_ids.push_back(1);
		//init_view_ids.push_back(2);
		//init_view_ids.push_back(3);

		int num_of_random_test = 1;

		bool do_ablation = false;
		bool eval_mode = false;

		for (int j = 0; j < method_ids.size(); j++) {
			for (int i = 0; i < names.size(); i++) {
				//保证5-60/100/144个视点有数据
				vector<View> init_views;
				shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1, method_ids[j]);
				shared_ptr<View_Planning_Simulator> simulator = make_shared<View_Planning_Simulator>(share_data);
				do_ablation = share_data->do_ablation;
				eval_mode = share_data->eval_mode;
				{//144
					int num_of_coverage_views = share_data->num_of_views;
					ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
					if (!fin_json.is_open()) {
						share_data->num_of_views = num_of_coverage_views;
						//read viewspace again
						ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
						share_data->pt_sphere.clear();
						share_data->pt_sphere.resize(share_data->num_of_views);
						for (int i = 0; i < share_data->num_of_views; i++) {
							share_data->pt_sphere[i].resize(3);
							for (int j = 0; j < 3; j++) {
								fin_sphere >> share_data->pt_sphere[i][j];
							}
						}
						cout << "view space size is: " << share_data->pt_sphere.size() << endl;
						Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
						share_data->pt_norm = pt0.norm();
						//reget viewspace
						simulator->view_space = make_shared<View_Space>(share_data);
						//get images
						simulator->get_coverage();
					}
				}
				{//5
					int num_of_coverage_views = 5;
					ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
					if (!fin_json.is_open()) {
						share_data->num_of_views = num_of_coverage_views;
						//read viewspace again
						ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
						share_data->pt_sphere.clear();
						share_data->pt_sphere.resize(share_data->num_of_views);
						for (int i = 0; i < share_data->num_of_views; i++) {
							share_data->pt_sphere[i].resize(3);
							for (int j = 0; j < 3; j++) {
								fin_sphere >> share_data->pt_sphere[i][j];
							}
						}
						cout << "view space size is: " << share_data->pt_sphere.size() << endl;
						Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
						share_data->pt_norm = pt0.norm();
						//reget viewspace
						simulator->view_space = make_shared<View_Space>(share_data);
						//get images
						simulator->get_coverage();
					}
				}
				cout << "init view space images." << endl;
				int num_of_views = share_data->num_of_views;
				for (int num_of_coverage_views = 5; num_of_coverage_views <= 60; num_of_coverage_views++) {
					ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
					if (!fin_json.is_open()) {
						share_data->num_of_views = num_of_coverage_views;
						//read viewspace again
						ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
						share_data->pt_sphere.clear();
						share_data->pt_sphere.resize(share_data->num_of_views);
						for (int i = 0; i < share_data->num_of_views; i++) {
							share_data->pt_sphere[i].resize(3);
							for (int j = 0; j < 3; j++) {
								fin_sphere >> share_data->pt_sphere[i][j];
							}
						}
						cout << "view space size is: " << share_data->pt_sphere.size() << endl;
						Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
						share_data->pt_norm = pt0.norm();
						//reget viewspace
						simulator->view_space = make_shared<View_Space>(share_data);
						//get images
						simulator->get_coverage();
					}
				}
				{//100
					int num_of_coverage_views = 100;
					ifstream fin_json(share_data->gt_path + "/" + to_string(num_of_coverage_views) + ".json");
					if (!fin_json.is_open()) {
						share_data->num_of_views = num_of_coverage_views;
						//read viewspace again
						ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
						share_data->pt_sphere.clear();
						share_data->pt_sphere.resize(share_data->num_of_views);
						for (int i = 0; i < share_data->num_of_views; i++) {
							share_data->pt_sphere[i].resize(3);
							for (int j = 0; j < 3; j++) {
								fin_sphere >> share_data->pt_sphere[i][j];
							}
						}
						cout << "view space size is: " << share_data->pt_sphere.size() << endl;
						Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
						share_data->pt_norm = pt0.norm();
						//reget viewspace
						simulator->view_space = make_shared<View_Space>(share_data);
						//get images
						simulator->get_coverage();
					}
				}
				{//5 init view
					share_data->num_of_views = 5;
					//read viewspace again
					ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
					share_data->pt_sphere.clear();
					share_data->pt_sphere.resize(share_data->num_of_views);
					for (int i = 0; i < share_data->num_of_views; i++) {
						share_data->pt_sphere[i].resize(3);
						for (int j = 0; j < 3; j++) {
							fin_sphere >> share_data->pt_sphere[i][j];
						}
					}
					cout << "view space size is: " << share_data->pt_sphere.size() << endl;
					Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
					share_data->pt_norm = pt0.norm();
					//reget viewspace
					simulator->view_space = make_shared<View_Space>(share_data);
					init_views = simulator->view_space->views;
					cout << "init view space with size: " << init_views.size() << endl;
				}
				//NBV测试
				for (int init_case = 0; init_case < init_view_ids.size(); init_case++) {
					for (int random_test_id = 0; random_test_id < num_of_random_test; random_test_id++) {
						//消融实验
						if (do_ablation) {
							for (int scop_min = 1; scop_min <= 8; scop_min += 1) {
								share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1, method_ids[j], scop_min);
								simulator = make_shared<View_Planning_Simulator>(share_data);
								simulator->init_views = init_views;
								//找到离init_view_ids[init_case]视点空间上最近的一个视点id
								int now_view_id = -1;
								double min_dis = 1e10;
								for (int i = 0; i < share_data->num_of_views; i++) {
									//计算两个视点之间的距离
									double dis = (simulator->view_space->views[i].init_pos - simulator->init_views[init_view_ids[init_case]].init_pos).norm();
									if (dis < min_dis) {
										min_dis = dis;
										now_view_id = i;
									}
								}
								if (now_view_id == -1) {
									cout << "can not find now view id" << endl;
								}
								share_data->start_view_id = now_view_id;
								cout << "start view planning" << endl;
								if (eval_mode) {
									//假设文件存在，只做最后的eval
									simulator->nbv_loop(now_view_id, init_view_ids[init_case], random_test_id, true);
								}
								else {
									//假设文件不存在，做所有的训练和测试
									simulator->nbv_loop(now_view_id, init_view_ids[init_case], random_test_id, false);
								}
							}
						}
						else {
							//正常跑
							share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1, method_ids[j]);
							simulator = make_shared<View_Planning_Simulator>(share_data);
							simulator->init_views = init_views;
							//找到离init_view_ids[init_case]视点空间上最近的一个视点id
							int now_view_id = -1;
							double min_dis = 1e10;
							for (int i = 0; i < share_data->num_of_views; i++) {
								//计算两个视点之间的距离
								double dis = (simulator->view_space->views[i].init_pos - simulator->init_views[init_view_ids[init_case]].init_pos).norm();
								if (dis < min_dis) {
									min_dis = dis;
									now_view_id = i;
								}
							}
							if (now_view_id == -1) {
								cout << "can not find now view id" << endl;
							}
							share_data->start_view_id = now_view_id;
							cout << "start view planning." << endl;
							simulator->nbv_loop(now_view_id, init_view_ids[init_case], random_test_id);
						}
					}
				}
			}
		}
	}
	else if (mode == RenderDiffusion) {
		int init_view_id = 0;
		for (int i = 0; i < names.size(); i++) {
			shared_ptr<Share_Data> share_data = make_shared<Share_Data>("../DefaultConfiguration.yaml", names[i], -1, -1, Imageto3DCovering);
			share_data->num_of_views = 5;
			ifstream fin_sphere(share_data->viewspace_path + to_string(share_data->num_of_views) + ".txt");
			share_data->pt_sphere.clear();
			share_data->pt_sphere.resize(share_data->num_of_views);
			for (int i = 0; i < share_data->num_of_views; i++) {
				share_data->pt_sphere[i].resize(3);
				for (int j = 0; j < 3; j++) {
					fin_sphere >> share_data->pt_sphere[i][j];
				}
			}
			cout << "view space size is: " << share_data->pt_sphere.size() << endl;
			Eigen::Vector3d pt0(share_data->pt_sphere[0][0], share_data->pt_sphere[0][1], share_data->pt_sphere[0][2]);
			share_data->pt_norm = pt0.norm();
			string pcd_path = share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/" + to_string(0) + ".pcd";
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
			if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcd_path, *temp_cloud) == -1) {
				cout << "Error: cannot read pcd file! " << pcd_path << endl;
				return -1;
			}
			share_data->cloud_ground_truth = temp_cloud;
			//reget viewspace
			View_Space render_vs(share_data);
			pcl::visualization::PCLVisualizer::Ptr viewer;
			viewer.reset(new pcl::visualization::PCLVisualizer("Render"));
			viewer->setBackgroundColor(255, 255, 255);
			viewer->initCameraParameters();
			viewer->addPointCloud<pcl::PointXYZRGB>(temp_cloud, "temp_cloud");
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, share_data->points_size_cloud, "temp_cloud");
			viewer->setSize(share_data->color_intrinsics.width, share_data->color_intrinsics.height);
			for (int i = 0; i < render_vs.views.size(); i++) {
				//获取视点位姿
				Eigen::Matrix4d view_pose_world;
				render_vs.views[i].get_next_camera_pos(share_data->now_camera_pose_world, share_data->object_center_world);
				view_pose_world = (share_data->now_camera_pose_world * render_vs.views[i].pose.inverse()).eval();
				//渲染
				Eigen::Matrix3f intrinsics;
				intrinsics << share_data->color_intrinsics.fx, 0, share_data->color_intrinsics.ppx,
					0, share_data->color_intrinsics.fy, share_data->color_intrinsics.ppy,
					0, 0, 1;
				Eigen::Matrix4f extrinsics = view_pose_world.cast<float>();
				viewer->setCameraParameters(intrinsics, extrinsics);
				viewer->spinOnce(100);
				share_data->access_directory(share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/" + to_string(share_data->num_of_views));
				viewer->saveScreenshot(share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/" + to_string(share_data->num_of_views) + "/rgb_" + to_string(i) + "_test.png");
				//注意pcl可能缩放了窗口，使用opencv检查图片
				cv::Mat img = cv::imread(share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/" + to_string(share_data->num_of_views) + "/rgb_" + to_string(i) + "_test.png");
				//检查share_data->color_intrinsics.width, share_data->color_intrinsics.height
				if (img.cols != share_data->color_intrinsics.width || img.rows != share_data->color_intrinsics.height) {
					//cout << i << " view-rendered image size is differnet. pick right-bottom block." << "\n";
					img = img(cv::Rect(img.cols - share_data->color_intrinsics.width,
						img.rows - share_data->color_intrinsics.height,
						share_data->color_intrinsics.width,
						share_data->color_intrinsics.height));
				}
				//cv::imwrite(share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/" + to_string(share_data->num_of_views) + "/rgb_" + to_string(i) + ".png", img);
				remove((share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/" + to_string(share_data->num_of_views) + + "/rgb_" + to_string(i) + "_test.png").c_str());
				cv::Mat rgb_image_alpha, rgb_image_alpha_clip;
				convertToAlpha(img, rgb_image_alpha);
				rgb_image_alpha_clip = rgb_image_alpha.clone();
				cv::flip(rgb_image_alpha_clip, rgb_image_alpha_clip, -1);
				cv::imwrite(share_data->one12345pp_path + share_data->name_of_pcd + "/" + to_string(init_view_id) + "/" + to_string(share_data->num_of_views) +"/rgbaClip_" + to_string(i) + ".png", rgb_image_alpha_clip);
			}
		}
	}
	cout << "System over." << endl;
	return 0;
}

/*
obj_000002
obj_000007
obj_000010
obj_000011
obj_000020
obj_000022
obj_000025
obj_000026
obj_000027
obj_000028
*/
