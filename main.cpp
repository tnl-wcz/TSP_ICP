#include <iostream>
#include <fstream>
#include "ICP.h"
#include "io_pc.h"
#include "TSP.h"
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <flann/flann.hpp>

#include <Eigen/SVD>
#include <iomanip>  // 添加头文件以使用std::fixed和std::setprecision
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <vector>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/random_sample.h>
#include <cmath>
#include <pcl/sample_consensus/sac_model_circle.h>
#include <opencv2/opencv.hpp>
#include <pcl/registration/icp.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>


typedef double Scalar;
typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Vertices;//Eigen矩阵类型，其大小是3xN，表示点云的顶点坐标。
typedef Eigen::Matrix<Scalar, 3, 1> VectorN;//Eigen矩阵类型，其大小是3x1，表示点的三维坐标。
typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> PointCloud_control;//读取控制点
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud_1;
typedef Eigen::Matrix<Scalar, 4, 4> Matrix4x4;
typedef Eigen::Transform<Scalar, 3, Eigen::Affine> AffineNd;

Vertices K;
Eigen::VectorXd line_direction(6);//垂直放置的对称轴

std::vector<Eigen::Vector3f> center_all;

// Convert Eigen::Matrix to PCL PointCloud
PointCloud_1::Ptr eigenToPointCloud(const Vertices& vertices) {
    PointCloud_1::Ptr cloud(new PointCloud_1);
    for (int i = 0; i < vertices.cols(); ++i) {
        pcl::PointXYZ point;
        point.x = vertices(0, i);
        point.y = vertices(1, i);
        point.z = vertices(2, i);
        cloud->points.push_back(point);
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

// Convert PCL PointCloud to Eigen::Matrix
Vertices pointCloudToEigen(const PointCloud_1::Ptr& cloud) {
    Vertices vertices(3, cloud->points.size());
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        vertices(0, i) = cloud->points[i].x;
        vertices(1, i) = cloud->points[i].y;
        vertices(2, i) = cloud->points[i].z;
    }
    return vertices;
}

// Function to filter a point cloud
Vertices filterPointCloud(const Vertices& vertices) {
    // Convert Eigen::Matrix to PCL PointCloud
    PointCloud_1::Ptr cloud = eigenToPointCloud(vertices);

    // Voxel Grid filter for downsampling
    pcl::VoxelGrid<pcl::PointXYZ> voxelGridFilter;
    voxelGridFilter.setInputCloud(cloud);
    voxelGridFilter.setLeafSize(0.1f, 0.1f, 0.1f); // Adjust the leaf size as needed
    PointCloud_1::Ptr cloudFiltered(new PointCloud_1);
    voxelGridFilter.filter(*cloudFiltered);

    // Statistical Outlier Removal filter for removing noise
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sorFilter;
    sorFilter.setInputCloud(cloudFiltered);
    sorFilter.setMeanK(500); // Number of neighbors to analyze for each point
    sorFilter.setStddevMulThresh(1.0); // Standard deviation multiplier threshold
    sorFilter.filter(*cloudFiltered);

    // Convert filtered PCL PointCloud back to Eigen::Matrix
    return pointCloudToEigen(cloudFiltered);
}


// 计算两个点之间的欧氏距离
double calculateDistance(const VectorN& p1, const VectorN& p2) {
    return (p1 - p2).norm();
}

PointCloud_control readcontrolPointCloudFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return PointCloud_control();
    }

    std::vector<Eigen::Vector4d> points;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double x, y, z;
        int index;
        if (!(iss >> x >> y >> z >> index)) {
            std::cerr << "Error: Invalid format in line: " << line << std::endl;
            continue;
        }
        points.push_back(Eigen::Vector4d(x, y, z, index));
    }

    file.close();

    // 将数据转换为 Eigen 的矩阵
    PointCloud_control PointCloud_control(3, points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        PointCloud_control.col(i) = points[i].head(3);
    }

    return PointCloud_control;
}

void scaleAndDeMeanPointCloud(PointCloud_control& vertices) {
    // 缩放
    VectorN point_cloud_scale;
    point_cloud_scale = vertices.rowwise().maxCoeff() - vertices.rowwise().minCoeff();
    double scale = point_cloud_scale.maxCoeff(); // 取最大的范围作为缩放因子
    vertices /= scale; // 将点云进行缩放

    // 去中心化
    VectorN point_cloud_mean;
    point_cloud_mean = vertices.rowwise().sum() / static_cast<double>(vertices.cols());
    vertices.colwise() -= point_cloud_mean; // 将每个点坐标减去均值
}

//计算中心点
Eigen::Vector3f computeMeanCenter(const std::vector<Eigen::Vector3f>& centers) {
    Eigen::Vector3f meanCenter(0.0f, 0.0f, 0.0f);

    if (!centers.empty()) {
        for (const auto& center : centers) {
            meanCenter += center;
        }

        meanCenter /= static_cast<float>(centers.size());
    }

    return meanCenter;
}


void fitCircle(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud,
               Eigen::Vector3f& center, float& radius) {
    // Extract 2D points (x, y)
    std::vector<cv::Point2f> points;
    for (const auto& point : cloud->points) {
        points.emplace_back(point.x, point.y);
    }

    // Fit a circle using OpenCV
    cv::Point2f center_cv;
    float radius_cv;
    cv::minEnclosingCircle(points, center_cv, radius_cv);

    // Convert the result to Eigen types
    center.x() = center_cv.x;
    center.y() = center_cv.y;
    center.z() = 0.0;  // Assuming the circle lies in the xy plane
    radius = radius_cv;

    center_all.push_back(center);
}




// 计算PCL点云在Z轴上的最大值和最小值的函数
void computeMinMaxZ(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& pointCloud, float& maxZ, float& minZ) {
    if (pointCloud->empty()) {
        std::cerr << "Error: Point cloud is empty." << std::endl;
        maxZ = minZ = 0.0; // 返回默认值，表示点云为空时的处理
        return;
    }

    maxZ = minZ = pointCloud->points[0].z; // 初始化为第一个点的Z值

    for (const auto& point : pointCloud->points) {
        if (point.z > maxZ) {
            maxZ = point.z;
        }
        if (point.z < minZ) {
            minZ = point.z;
        }
    }
}


// 截取Z轴上每隔20的点云段的函数
void extractSegments(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& inputCloud, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& segments) {
    segments.clear(); // 清空存储段的vector

    if (inputCloud->empty()) {
        std::cerr << "Error: Input point cloud is empty." << std::endl;
        return;
    }

    float minZ, maxZ;
    computeMinMaxZ(inputCloud, maxZ, minZ); // 使用前面提到的计算最小和最大Z值的函数

    const float segmentSize = 1; // 每隔20的Z轴段

    int currentZ = (int)(maxZ - minZ)/2 + minZ;
    int segmentIndex = 1;
    while (currentZ <= maxZ - (maxZ - minZ)/20) {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr segment(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

        // 提取当前Z轴段的点
        pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
        pcl::PointIndices::Ptr indices(new pcl::PointIndices);
        for (std::size_t i = 0; i < inputCloud->size(); ++i) {
            if (inputCloud->points[i].z >= currentZ && inputCloud->points[i].z < currentZ + segmentSize) {
                indices->indices.push_back(static_cast<int>(i));
            }
        }

        extract.setInputCloud(inputCloud);
        extract.setIndices(indices);
        extract.filter(*segment);

        // std::ofstream outputFile;
        // std::string segmentFilename = "/home/wcz/ceres_ws/src/axis/data/xhp/crap/" + std::to_string(segmentIndex) + ".txt";
        // outputFile.open(segmentFilename);
        
        if(inputCloud->size() > 0){
            ++segmentIndex;
        // 提取并保存当前Z轴段的点
        // for (std::size_t i = 0; i < inputCloud->size(); ++i) {
            
        //     if (inputCloud->points[i].z >= currentZ && inputCloud->points[i].z < currentZ + segmentSize) {
        //         outputFile << inputCloud->points[i].x << " " << inputCloud->points[i].y << " " << inputCloud->points[i].z << " " <<
        //         inputCloud->points[i].normal_x << " " << inputCloud->points[i].normal_y << " " << inputCloud->points[i].normal_z << "\n";

          
        //     }
        // }

        pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);
        for(std::size_t i = 0; i < segment->size(); ++i)
        {
            pcl::PointXYZ point;
            point.x = segment->points[i].x;
            point.y = segment->points[i].y;
            point.z = segment->points[i].z;
            pointCloud->push_back(point);
        }
 

        // // 关闭文件
        // outputFile.close();


        // 将当前段添加到结果中
        segments.push_back(segment);
        }


        // 移动到下一个Z轴段
        currentZ += 1;
        
    }
}
// 计算向量与Z轴的偏差角
float computeDeviationFromZAxis_angle(const Eigen::Vector3f& vector) {
    // 计算向量与Z轴的夹角（偏差角）
    float angle = std::acos(vector.normalized().dot(Eigen::Vector3f::UnitZ()));
    return angle;
}

void projectPointsToXOY(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& segment) {
    // 实现将点云段投影到XOY平面的代码
    for (auto& point : segment->points) {
        point.z = 0.0;  // Set the z-coordinate to 0 to project to XOY plane
    }

}

void saveSegmentsToTxtFiles(const std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& segments,
                             const std::string& basePath) {
    for (std::size_t i = 0; i < segments.size(); ++i) {
        std::ofstream outputFile;
        std::string segmentFilename = basePath + std::to_string(i + 1) + ".txt";
        outputFile.open(segmentFilename);

        if (outputFile.is_open()) {
            for (std::size_t j = 0; j < segments[i]->size(); ++j) {
                outputFile << segments[i]->points[j].x << " " << segments[i]->points[j].y << " " 
                           << segments[i]->points[j].z << " " << segments[i]->points[j].normal_x
                           << " " << segments[i]->points[j].normal_y << " " << segments[i]->points[j].normal_z << "\n";
            }

            // 关闭文件
            outputFile.close();
        } else {
            std::cerr << "Error opening file: " << segmentFilename << std::endl;
        }
    }
}

// 计算向量与Z轴的偏差
float computeDeviationFromZAxis(const Eigen::Vector3f& vector) {
    // 计算向量与Z轴的夹角（偏差）
    float angle = std::acos(vector.normalized().dot(Eigen::Vector3f::UnitZ()));
    return angle;
}

Eigen::Vector3f computeMean(const std::vector<Eigen::Vector3f>& vectors) {
    if (vectors.empty()) {
        std::cerr << "Input vector container is empty." << std::endl;
        return Eigen::Vector3f::Zero();
    }

    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    for (const auto& vector : vectors) {
        sum += vector;
    }

    return sum / static_cast<float>(vectors.size());
}

// 选择偏差最小的向量
Eigen::Vector3f chooseVectorWithMinDeviation(const std::vector<Eigen::Vector3f>& vectors) {
    if (vectors.empty()) {
        std::cerr << "Input vector container is empty." << std::endl;
        return Eigen::Vector3f::Zero();
    }

    float minDeviation = std::numeric_limits<float>::infinity();
    Eigen::Vector3f selectedVector;

    // for (const auto& vector : vectors) {
    //     float deviation = computeDeviationFromZAxis(vector);
    //     if (deviation < minDeviation) {
    //         minDeviation = deviation;
    //         selectedVector = vector;
    //     }
    // }
    selectedVector = computeMean(vectors);
    return selectedVector;
}

//计算外积
Eigen::Vector3f computePerpendicularVector(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud) {
    // 存储每次迭代的结果的容器
    std::vector<Eigen::Vector3f> results;
    // 迭代次数
    const int numIterations = 10;

    for (int iteration = 0; iteration < numIterations; ++iteration) {
    // 确保点云中有足够的点
    if (cloud->size() < 2) {
        std::cerr << "Not enough points in the cloud." << std::endl;
        return Eigen::Vector3f::Zero();
    }

    // 从点云中随机抽取两个点的索引
    std::vector<int> indices(4); // 初始化大小为2
    pcl::RandomSample<pcl::PointXYZRGBNormal> randomSample;
    randomSample.setInputCloud(cloud);
    randomSample.setSample(4); // 设置抽取2个样本
    randomSample.filter(indices);

    if (indices.size() < 2) {
        std::cerr << "Failed to generate random indices." << std::endl;
        return Eigen::Vector3f::Zero();
    }

    // 计算两个点的向量
    Eigen::Vector3f vectorAB(cloud->points[indices[1]].x - cloud->points[indices[0]].x,
                cloud->points[indices[1]].y - cloud->points[indices[0]].y,
                cloud->points[indices[1]].z - cloud->points[indices[0]].z);

        // 计算两个点的向量
    Eigen::Vector3f vectorCD(cloud->points[indices[3]].x - cloud->points[indices[2]].x,
                cloud->points[indices[3]].y - cloud->points[indices[2]].y,
                cloud->points[indices[3]].z - cloud->points[indices[2]].z);

    //   // 求垂直于向量AB的向量，可以采用叉乘的方式
    //   Eigen::Vector3f perpendicularVector = vectorAB.cross(Eigen::Vector3f::UnitZ());
        // 计算向量 vectorAB 和 vectorCD 的外积（叉乘）
        Eigen::Vector3f vectorCrossProduct = vectorAB.cross(vectorCD);


    // 检查是否得到零向量，如果是，则尝试与其他单位向量叉乘
    if (vectorCrossProduct.norm() < 1e-6) {
        vectorCrossProduct = vectorAB.cross(Eigen::Vector3f::UnitX());
    }

    // 存储每次迭代的结果
    results.push_back(vectorCrossProduct.normalized());

    }
    // 选择偏差最小的向量
    Eigen::Vector3f selectedVector = chooseVectorWithMinDeviation(results);


    return selectedVector.normalized();
}

void calculateMaxOffsetsAndAngles(const std::vector<Eigen::Vector3f>& vectors, 
                                  float& maxPositiveOffset, float& maxNegativeOffset,
                                  float& maxPositiveAngle, float& maxNegativeAngle) {
    maxPositiveOffset = 0.0f;
    maxNegativeOffset = 0.0f;
    maxPositiveAngle = 0.0f;
    maxNegativeAngle = 0.0f;

    float mpoffset = 0.0f;
    float mnoffset = 0.0f;

    for (const auto& vector : vectors) {
        // 计算向量在 X 轴上的投影
        float xProjection = vector.x();

        // 更新正负偏移量
        mpoffset = std::max(mpoffset, xProjection);
        mnoffset = std::min(mnoffset, xProjection);

        // 仅在偏差小于0.1时才更新正负角度
        if (std::abs(xProjection) < 0.1) {
            // 转换偏移量为角度
            float angle = std::atan(std::abs(xProjection) / vector.norm()) * 180.0f / M_PI;

            if (xProjection > 0) {
                maxPositiveAngle = std::max(maxPositiveAngle, angle);
            } else {
                maxNegativeAngle = std::max(maxNegativeAngle, angle);
            }
        }
    }

    // 将最终的正负偏移量赋给传入的参数
    maxPositiveOffset = std::max(mpoffset, 0.0f);  // 确保偏移量非负
    maxNegativeOffset = std::min(mnoffset, 0.0f);  // 确保偏移量非正
}

Eigen::Vector3f flipAndComputeMean(const std::vector<Eigen::Vector3f>& linevector) {
    Eigen::Vector3f mean(0.0f, 0.0f, 0.0f);

    for (const auto& vec : linevector) {
        Eigen::Vector3f positive_z = vec;
        if (vec(2) < 0) {
            positive_z(2) = -positive_z(2);
        }
        mean += positive_z;
    }

    // Calculate mean
    mean /= linevector.size();

    return mean;
}

// 直线均值填充函数
void fillDirection(Eigen::VectorXd& line_direction, Eigen::Vector3f& meanCenter,Eigen::Vector3f& meanVector) {
    // 检查 line_direction 的维度是否满足要求
    assert(line_direction.size() >= 3);

    // 将 meanCenter 的元素填充到 line_direction 的前三个元素
    line_direction[0] = meanCenter[0];
    line_direction[1] = meanCenter[1];
    line_direction[2] = meanCenter[2];
    line_direction[3] = meanVector[0];
    line_direction[4] = meanVector[1];
    line_direction[5] = meanVector[2];
}

void CalculateAxis(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& sourcepoints)
{
    // pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sourcepoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    // loadPointsFromTxt("/home/wcz/ceres_ws/src/axis/data/xhp/xhps1.txt", sourcepoints);

    // 计算最大Z值和最小Z值
    float maxZ, minZ;
    computeMinMaxZ(sourcepoints, maxZ, minZ);
    // 打印结果
    std::cout << "Max Z value in the point cloud: " << maxZ << std::endl;
    std::cout << "Min Z value in the point cloud: " << minZ << std::endl;


    // 截取Z轴上每隔20的点云段
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> segments;
    extractSegments(sourcepoints, segments);

    // for (auto& segment : segments) {
    //     projectPointsToXOY(segment);
    // }


    // saveSegmentsToTxtFiles(segments, basePath);

    for(auto i = 0;i < segments.size();i++)
    {
    // 假设您的点云已经被填充到了segments[0]中
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud = segments[i];

    Eigen::Vector3f center;
    float radius;
    // std::cout << "segments:" << i+1 << std::endl;
    if(cloud->size()>0){
        fitCircle(cloud, center, radius);
    }
    
    }
    std::cout << "segments:" << segments.size() << std::endl;

    // for(auto center:center_all)
    // {
    //     std::cout << center.transpose() << std::endl;
    // }
    // 计算均值圆心
    Eigen::Vector3f meanCenter = computeMeanCenter(center_all);

    // 打印结果
    std::cout << "Mean Center: (" << meanCenter.x() << ", " << meanCenter.y() << ", " << meanCenter.z() << ")" << std::endl;

    std::vector<Eigen::Vector3f> linevector;
    std::vector<Eigen::Vector3f> linevector_XOZ;
    int verctor_count = 0;
    for(auto i = 0 ;i < segments.size(); i++)
    {
        // 计算垂直于两个随机点向量的向量
        Eigen::Vector3f perpendicularVector = computePerpendicularVector(segments[i]);
        if(std::abs(perpendicularVector.z()) > 0.999){
        linevector.push_back(perpendicularVector);

        // 将向量投影到XZ平面（将Y分量设为零）
        perpendicularVector.y() = 0.0f;

        linevector_XOZ.push_back(perpendicularVector);
        verctor_count++;
        }
        // 打印投影后的结果
        
    }
    std::cout << "vector z > 0.999:  " << verctor_count << std::endl;
    // 计算 X 轴上的正负最大偏移量和转换为角度
    float maxPositiveOffset, maxNegativeOffset, maxPositiveAngle, maxNegativeAngle;
    calculateMaxOffsetsAndAngles(linevector_XOZ, maxPositiveOffset, maxNegativeOffset, maxPositiveAngle, maxNegativeAngle);

    // 打印结果
    std::cout << "Max Positive Offset on X Axis: " << maxPositiveOffset << std::endl;
    std::cout << "Max Positive Angle: " << maxPositiveAngle << " degrees" << std::endl;
    std::cout << "Max Negative Offset on X Axis: " << maxNegativeOffset << std::endl;
    std::cout << "Max Negative Angle: " << maxNegativeAngle << " degrees" << std::endl;

    // Call the function
    Eigen::Vector3f meanVector = flipAndComputeMean(linevector);

    // Output the mean vector
    std::cout << "Mean Vector: " << meanVector.transpose() << std::endl;

    // 计算向量与Z轴的偏差角
    float deviationAngle = computeDeviationFromZAxis_angle(meanVector);

    // 打印结果
    std::cout << "Deviation Angle from Z Axis: " << deviationAngle << " radians" << std::endl;

    fillDirection(line_direction,meanCenter,meanVector);

}

void writeVerticesToFile(const Vertices& vertices, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        // Write each column (each 3D point) to the file
        for (int i = 0; i < vertices.cols(); ++i) {
            for (int j = 0; j < vertices.rows(); ++j) {
                outputFile << vertices(j, i) << " ";
            }
            outputFile << std::endl;
        }

        std::cout << "Vertices have been written to " << filename << std::endl;
        outputFile.close();
    } else {
        std::cerr << "Unable to open the file: " << filename << std::endl;
    }
}

void computeZMinMax(const Vertices& points, double& zMin, double& zMax) {
    // 初始化最小值和最大值
    zMin = std::numeric_limits<double>::max();
    zMax = -std::numeric_limits<double>::max();

    // 遍历每一列
    for (int i = 0; i < points.cols(); ++i) {
        double z = points(2, i); // 获取Z坐标

        // 更新最小值和最大值
        if (z < zMin) {
            zMin = z;
        }
        if (z > zMax) {
            zMax = z;
        }
    }
}

void filterPointsByZRange(const Vertices& points, double zMin, double zMax, Vertices& filteredPoints) {
    // 初始化输出矩阵
    filteredPoints.resize(3, 0);

    // 遍历每一列
    for (int i = 0; i < points.cols(); ++i) {
        double z = points(2, i); // 获取Z坐标

        double k = (zMax - zMin)/3.0;


        // 检查是否在指定范围内
        if (z >= zMin + k && z <= zMax - k) {
            // 将满足条件的点添加到输出矩阵中
            filteredPoints.conservativeResize(3, filteredPoints.cols() + 1);
            filteredPoints.col(filteredPoints.cols() - 1) = points.col(i);
        }
    }
}
double point_to_line_perpendicular_distance(const Eigen::VectorXd& pointX, const Eigen::Vector3d& pointY, const Eigen::VectorXd& line_direction) {
    // 计算直线上的一点
    Eigen::VectorXd point_on_line = pointY;

    // 计算点X到直线上点的向量
    Eigen::VectorXd point_to_line_vector = pointX - point_on_line;

    // 计算点到直线的投影向量
    Eigen::VectorXd projection = point_to_line_vector - (point_to_line_vector.dot(line_direction.segment(0, 3)) / line_direction.segment(0, 3).squaredNorm()) * line_direction.segment(0, 3);

    // 计算投影向量的模长，即点到直线的垂直距离
    double perpendicular_distance = projection.norm();

    return perpendicular_distance;
}

// 寻找源点云 X 到目标点云 Y 的最优刚性变换
void find_nearest_neighbors(const Vertices& X, const Vertices& Y, Eigen::MatrixXi& indices, Eigen::MatrixXd& distances, double threshold, Vertices& Q, Vertices& K) {
    const size_t N = X.cols();
    const size_t M = Y.cols();

    // 构建目标点云
    flann::Matrix<float> dataset(new float[M * 3], M, 3);
    for (size_t i = 0; i < M; ++i) {
        dataset[i][0] = Y(0, i);
        dataset[i][1] = Y(1, i);
        dataset[i][2] = Y(2, i);
    }

    flann::Index<flann::L2<float>> index(dataset, flann::KDTreeSingleIndexParams(10));
    index.buildIndex();

    // 查找每个源点云 X 中点在目标点云 Y 中的最近邻
    flann::Matrix<int> flann_indices(new int[N], N, 1);
    flann::Matrix<float> flann_dists(new float[N], N, 1);

    flann::Matrix<float> query(new float[N * 3], N, 3);
    for (size_t i = 0; i < N; ++i) {
        query[i][0] = X(0, i);
        query[i][1] = X(1, i);
        query[i][2] = X(2, i);
    }

    indices.resize(N, 1);
    distances.resize(N, 1);

    index.knnSearch(query, flann_indices, flann_dists, 1, flann::SearchParams(10));

    // 将最近邻点复制到矩阵 Q 中，并保存到 K 中
    Q.resize(3, N);
    K.resize(3, N);
    size_t validPointCount = 0;

    for (size_t i = 0; i < N; ++i) {
        indices(i, 0) = flann_indices[i][0];
        distances(i, 0) = flann_dists[i][0];

        // 如果距离超过阈值，不保存该点
        if (distances(i, 0) <= threshold) {
            Q(0, validPointCount) = Y(0, indices(i, 0));
            Q(1, validPointCount) = Y(1, indices(i, 0));
            Q(2, validPointCount) = Y(2, indices(i, 0));
            K(0, validPointCount) = X(0, i);
            K(1, validPointCount) = X(1, i);
            K(2, validPointCount) = X(2, i);
            validPointCount++;
        }
    }

    // 重新调整矩阵大小，仅保留有效点云
    Q.conservativeResize(3, validPointCount);
    K.conservativeResize(3, validPointCount);

    delete[] dataset.ptr();
    delete[] query.ptr();
}

// 静态成员函数
template <typename T>
static T pointToLineDistance(const Eigen::Matrix<T, 3, 1>& point,
                const Eigen::Matrix<T, 3, 1>& linePoint,
                const Eigen::Matrix<T, 3, 1>& lineNormal) {
    Eigen::Matrix<T, 3, 1> pointToLine = point - linePoint;
    T distance = pointToLine.dot(lineNormal) / lineNormal.norm();
    // return ceres::sqrt(ceres::abs(distance)); // Using ceres::sqrt and ceres::abs for compatibility with ceres::Jet.
    return distance; // Return the raw distance value.
}

//ceres优化部分
struct PointToPointCostFunction {
PointToPointCostFunction(const double& sx, const double& sy, const double& sz,
                        const double& tx, const double& ty, const double& tz,
                        const Eigen::VectorXd& line_direction)
    : source_(sx, sy, sz),  // 创建源点的3D向量
        target_(tx, ty, tz),  // 创建目标点的3D向量
        line_direction_(line_direction.normalized()){}

    template <typename T>
    bool operator()(const T* const parameters, T* residuals) const {
        // Extract parameters
        const T& Tx = parameters[0];
        const T& Ty = parameters[1];
        const T& Tz = parameters[2];
        const T* quaternion = parameters + 3;

        // Transform source point
        Eigen::Matrix<T, 3, 1> sourcePoint;
        Eigen::Matrix<T, 3, 1> targetPoint;
        Eigen::Matrix<T, 3, 1> targetPoint_trans;

        sourcePoint << T(source_(0)), T(source_(1)), T(source_(2));
        targetPoint << T(target_(0)), T(target_(1)), T(target_(2));

        // Construct transformation matrix
        Eigen::Transform<T, 3, Eigen::Affine> transform_matrix;
        transform_matrix.translation() << Tx, Ty, Tz;
        transform_matrix.linear() = Eigen::Quaternion<T>(quaternion).toRotationMatrix();
        // std::cout << "Transformation matrix:\n" << transform_matrix.matrix() << std::endl;
        
        targetPoint_trans = transform_matrix * sourcePoint;
        // std::cout << "targetPoint_trans: " << targetPoint_trans << std::endl;
        // std::cout << "targetPoint: " << targetPoint << std::endl;

        Eigen::Matrix<T, 3, 1> line_Point = line_direction_.template cast<T>().head(3);
        Eigen::Matrix<T, 3, 1> line_Normal = line_direction_.template cast<T>().tail(3);
        Eigen::Matrix<T, 3, 1> sourcePoint3D = sourcePoint.template head<3>().template cast<T>();
        Eigen::Matrix<T, 3, 1> targetPoint3D = targetPoint.template head<3>().template cast<T>();
        T distance_source = pointToLineDistance(targetPoint_trans, line_Point, line_Normal);
        T distance_target = pointToLineDistance(targetPoint3D, line_Point, line_Normal);

        // Compute the residual as the difference in distances
        residuals[0] = abs(distance_target - distance_source);
        // std::cout << "distance_source: " << distance_source << std::endl;
        // std::cout << "distance_target: " << distance_target << std::endl;
        // std::cout << "residuals: " << residuals[0] << std::endl;
    return true;
}

private:
const Eigen::Vector3d source_;
const Eigen::Vector3d target_;
Eigen::VectorXd line_direction_;
};

Vertices merge_cloud(const Vertices& vertices_source, const Vertices& vertices_target) {
    // 创建一个新的矩阵，大小为两个输入矩阵的列之和
    Vertices merge_cloud(3, vertices_source.cols() + vertices_target.cols());

    // 将源矩阵的数据复制到新矩阵
    merge_cloud.block(0, 0, 3, vertices_source.cols()) = vertices_source;

    // 将目标矩阵的数据复制到新矩阵的适当位置
    merge_cloud.block(0, vertices_source.cols(), 3, vertices_target.cols()) = vertices_target;

    return merge_cloud;
}

double calculate_rmse(const Vertices& Q, const Vertices& K) {
    // 检查点云的列数是否相同
    if (Q.cols() != K.cols()) {
        std::cerr << "Error: Number of points in point clouds are not equal!" << std::endl;
        return -1.0; // 返回错误码
    }

    // 计算点云的数量
    size_t num_points = Q.cols();

    // 初始化平方误差和
    double sum_squared_error = 0.0;

    // 循环遍历每个点，计算距离的平方误差和
    for (size_t i = 0; i < num_points; ++i) {
        // 获取第 i 个点的坐标
        VectorN point_Q = Q.col(i);
        VectorN point_K = K.col(i);

        // 计算点之间的欧氏距离的平方，并将其添加到误差和中
        double squared_error = (point_Q - point_K).squaredNorm();
        sum_squared_error += squared_error;
    }

    // 计算均方根误差(RMSE)
    double rmse = std::sqrt(sum_squared_error / static_cast<double>(num_points));

    return rmse;
}

// 坐标变换函数
PointCloud_control transformPointCloud(const PointCloud_control& pointCloud, const Matrix4x4& transformation) {
    // 将点云转换为齐次坐标表示
    Eigen::Matrix<Scalar, 4, Eigen::Dynamic> homogenousPoints(4, pointCloud.cols());
    homogenousPoints.topRows(3) = pointCloud;
    homogenousPoints.row(3).setOnes();

    // 应用变换
    homogenousPoints = transformation * homogenousPoints;

    // 转换回非齐次坐标表示
    PointCloud_control transformedPointCloud(3, pointCloud.cols());
    transformedPointCloud = homogenousPoints.topRows(3);

    return transformedPointCloud;
}

// 计算两个点之间的欧几里得距离
double euclidean_distance(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
    return (p2 - p1).norm(); // 使用 Eigen 的 norm 函数计算向量的范数
}

void findNeighborsWithinRadius(const Vertices& pointCloud_s1_tie, const Vertices& vertices_target_input, double radius, Vertices& p1) {
    const size_t N = pointCloud_s1_tie.cols();
    const size_t M = vertices_target_input.cols();

    // 构建目标点云
    flann::Matrix<float> dataset(new float[M * 3], M, 3);
    for (size_t i = 0; i < M; ++i) {
        dataset[i][0] = vertices_target_input(0, i);
        dataset[i][1] = vertices_target_input(1, i);
        dataset[i][2] = vertices_target_input(2, i);
    }

    // 构建 KD 树
    flann::Index<flann::L2<float>> index(dataset, flann::KDTreeSingleIndexParams(10));
    try {
        index.buildIndex();
    } catch (flann::FLANNException& e) {
        std::cerr << "Error building FLANN index: " << e.what() << std::endl;
        delete[] dataset.ptr();
        return;
    }

    // 查询每个源点的半径范围内的邻近点
    flann::Matrix<int> flann_indices(new int[N * 2000], N, 2000);
    flann::Matrix<float> flann_dists(new float[N * 2000], N, 2000);

    flann::Matrix<float> query(new float[N * 3], N, 3);
    for (size_t i = 0; i < N; ++i) {
        query[i][0] = pointCloud_s1_tie(0, i);
        query[i][1] = pointCloud_s1_tie(1, i);
        query[i][2] = pointCloud_s1_tie(2, i);
    }

    flann::SearchParams params(-1); // 无限搜索
    try {
        index.radiusSearch(query, flann_indices, flann_dists, radius * radius, params);
    } catch (flann::FLANNException& e) {
        std::cerr << "Error during radius search: " << e.what() << std::endl;
        delete[] flann_indices.ptr();
        delete[] flann_dists.ptr();
        delete[] query.ptr();
        delete[] dataset.ptr();
        return;
    }

    // 计算总的邻近点数量
    size_t total_neighbors = 2000 * N;

    // 动态调整 p1 的大小以容纳所有的邻近点
    p1.resize(3, total_neighbors);

    size_t count = 0;
    for (size_t i = 0; i < N; ++i) {
        const size_t num_neighbors = flann_indices[i][0]; // 获取找到的邻近点数量

        // 将半径范围内的邻近点的坐标保存到 p1 中
        for (size_t j = 0; j < num_neighbors; ++j) {
            if (count >= total_neighbors) {
                std::cerr << "Error: Insufficient memory allocated for storing neighbors." << std::endl;
                return; // 停止函数执行并打印错误消息
            }
            const int neighbor_index = flann_indices[i][j + 1]; // 注意索引从0开始
            Eigen::Map<const Eigen::Vector3f> neighbor(&dataset[neighbor_index][0]);
            p1.col(count++) = neighbor.cast<Scalar>(); // 将 float 转换为 Scalar
        }
    }

    // 调整矩阵大小，删除多余的列
    p1.conservativeResize(3, count);

    delete[] flann_indices.ptr();
    delete[] flann_dists.ptr();
    delete[] query.ptr();
    delete[] dataset.ptr();
}

// 计算点到给定点的距离
std::vector<std::pair<double, int>> computeDistances(const Vertices& pointCloud, const Eigen::Vector3d& queryPoint) {
    std::vector<std::pair<double, int>> distances;
    for (int i = 0; i < pointCloud.cols(); ++i) {
        Eigen::Vector3d point = pointCloud.col(i);
        double distance = (point - queryPoint).norm();
        distances.push_back(std::make_pair(distance, i)); // 存储距离和点的索引
    }
    return distances;
}

// 找到离给定点最近的500个点
Vertices findNearestPoints(const Vertices& pointCloud, const Eigen::Vector3d& queryPoint, int numNeighbors) {
    std::vector<std::pair<double, int>> distances = computeDistances(pointCloud, queryPoint);
    std::sort(distances.begin(), distances.end());

    Vertices nearestPoints(3, numNeighbors);
    for (int i = 0; i < numNeighbors; ++i) {
        nearestPoints.col(i) = pointCloud.col(distances[i].second);
    }
    return nearestPoints;
}


// 合并多个 Vertices 到一个 Vertices 中
Vertices mergeVertices(const std::vector<Vertices>& vertices_list) {
    // 计算所有点的总数
    size_t total_points = 0;
    for (const auto& vertices : vertices_list) {
        total_points += vertices.cols();
    }

    // 创建新的 Vertices
    Vertices merged_vertices(3, total_points);

    // 将每个 Vertices 中的点添加到新的 Vertices 中
    size_t current_index = 0;
    for (const auto& vertices : vertices_list) {
        merged_vertices.block(0, current_index, 3, vertices.cols()) = vertices;
        current_index += vertices.cols();
    }

    return merged_vertices;
}

void convertToPclPointCloud(const Vertices& vertices, PointCloud_1::Ptr& pclPointCloud) {
    pclPointCloud->clear(); // 清空PointCloud对象

    // 遍历自定义点云数据，并将其复制到PointCloud对象中
    for (int i = 0; i < vertices.cols(); ++i) {
        pcl::PointXYZ point;
        point.x = vertices(0, i);
        point.y = vertices(1, i);
        point.z = vertices(2, i);
        pclPointCloud->push_back(point);
    }
}
// 定义点云转换函数
Vertices pclPointCloudToVertices(const PointCloud_1::Ptr& pcl_cloud)
{
    Vertices vertices(3, pcl_cloud->size());

    for (size_t i = 0; i < pcl_cloud->size(); ++i)
    {
        vertices(0, i) = pcl_cloud->points[i].x;
        vertices(1, i) = pcl_cloud->points[i].y;
        vertices(2, i) = pcl_cloud->points[i].z;
    }

    return vertices;
}

Vertices ICP_pcl(Vertices vertices_source,Vertices vertices_target,Eigen::Matrix4f& transformation)
{
    // 创建PCL的PointCloud对象
    PointCloud_1::Ptr pcl_source(new PointCloud_1);
    PointCloud_1::Ptr pcl_target(new PointCloud_1);

    // 转换自定义的点云格式到PCL的PointCloud格式
    convertToPclPointCloud(vertices_source, pcl_source);
    convertToPclPointCloud(vertices_target, pcl_target);

    // 使用ICP算法
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(pcl_source);
    icp.setInputTarget(pcl_target);
    PointCloud_1::Ptr aligned_cloud(new PointCloud_1); // 存储配准后的点云

    // 设置ICP参数
    icp.setMaximumIterations(500);
    icp.setTransformationEpsilon(1e-8);
    icp.setMaxCorrespondenceDistance(0.01);

    // 执行ICP算法
    PointCloud_1::Ptr aligned(new PointCloud_1);
    icp.align(*aligned);

    // 获取配准变换矩阵
    transformation = icp.getFinalTransformation();

    // 将源点云应用配准变换
    pcl::transformPointCloud(*pcl_source, *aligned_cloud, transformation);

    // 将配准后的点云转换为Vertices格式
    Vertices aligned_vertices = pclPointCloudToVertices(aligned_cloud);
    return aligned_vertices;

}

double calculateMeanDistance(const AffineNd& T, const PointCloud_control& pointCloud_control, const PointCloud_control& pointCloud_S2,
                                    const std::vector<std::vector<double>>& distance_pointcloud_control) 
{
        PointCloud_control pointCloud_s2 = T * pointCloud_S2;

        std::vector<std::vector<double>> distance_pointcloud_control_T;
        std::vector<double> distances_each;
        for (int i = 0; i < pointCloud_control.cols(); ++i) {
            VectorN point_control = pointCloud_control.col(i);
            for(int j = 0; j < pointCloud_s2.cols();j++)
            {
                VectorN Point = pointCloud_s2.col(j);
                double distance = calculateDistance(point_control, Point);
                distances_each.push_back(distance);
            }

            distance_pointcloud_control_T.push_back(distances_each);  
        }

        std::cout << distance_pointcloud_control_T.size() << "\n" << std::endl;
        std::cout << distance_pointcloud_control_T[0].size() << "\n" << std::endl;
        double distance_control;
        std::vector<double> distances_control_L;
        for(int i = 0;i < distance_pointcloud_control.size();i++)
        {
            for(int j = 0;j < distance_pointcloud_control[i].size();j++)
            {
                distance_control = std::abs(distance_pointcloud_control[i][j] - distance_pointcloud_control_T[i][j]);
                // std::cout << distance_pointcloud_control[i][j] << std::endl;
                // std::cout << distance_pointcloud_control_T[i][j] << std::endl;
                // std::cout << distance_control<< "\n" << std::endl;
                distances_control_L.push_back(distance_control);
            }
        }
        double sum = 0;
        for(int i = 0;i < distances_control_L.size();i++)
        {
            sum += distances_control_L[i];
        }
        std::cout << sum << std::endl;
        std::cout << distances_control_L.size() << std::endl;
        double mean = sum/distances_control_L.size();

        return mean;
}

double calculateRMSE(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("两个向量必须具有相同的大小。");
    }

    double sumSquaredError = 0.0;

    for (size_t i = 0; i < vec1.size(); ++i) {
        double diff = vec1[i] - vec2[i];
        sumSquaredError += diff * diff;
    }

    double rmse = std::sqrt(sumSquaredError / vec1.size());
    return rmse;
}




int main(int argc, char const ** argv)
{
    std::string file_source;
    std::string file_target;
    std::string file_init;
    std::string res_trans_path;
    std::string out_path;
    bool use_init = true;
    MatrixXX res_trans;
    enum Method{ICP, AA_ICP, FICP, RICP, PPL, RPPL, SparseICP, SICPPPL} method=RICP;
    if(argc == 5)
    {
        file_target = argv[1];
        file_source = argv[2];
        out_path = argv[3];
        method = Method(std::stoi(argv[4]));
    }
    else if(argc==4)
    {
        file_target = argv[1];
        file_source = argv[2];
        out_path = argv[3];
    }
    else
    {
        std::cout << "Usage: target.ply source.ply out_path <Method>" << std::endl;
        std::cout << "Method :\n"
                  << "0: ICP\n1: AA-ICP\n2: Our Fast ICP\n3: Our Robust ICP\n4: ICP Point-to-plane\n"
                  << "5: Our Robust ICP point to plane\n6: Sparse ICP\n7: Sparse ICP point to plane" << std::endl;
        exit(0);
    }
    int dim = 3;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sourcepoints(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    // 从.ply文件中读取点云数据
    if (pcl::io::loadPLYFile<pcl::PointXYZRGBNormal>(file_target, *sourcepoints) == -1) {
        PCL_ERROR("Couldn't read file.\n");
        return -1;
    }
    CalculateAxis(sourcepoints);


    //--- Model that will be rigidly transformed
    Vertices vertices_source, normal_source, src_vert_colors;
    read_file(vertices_source, normal_source, src_vert_colors, file_source);//通过 read_file 函数读取源点云和目标点云的顶点坐标。
    Vertices vertices_source_input = vertices_source;
    std::cout << "source: " << vertices_source.rows() << "x" << vertices_source.cols() << std::endl;

    //--- Model that source will be aligned to
    Vertices vertices_target, normal_target, tar_vert_colors;
    read_file(vertices_target, normal_target, tar_vert_colors, file_target);
    Vertices vertices_target_input = vertices_target;
    double zMin, zMax;
    computeZMinMax(vertices_target, zMin, zMax);

    // Filter the point cloud
    // vertices_source = filterPointCloud(vertices_source);
    // vertices_target = filterPointCloud(vertices_target);


    // 输出最小值和最大值
    std::cout << "Minimum Z: " << zMin << std::endl;
    std::cout << "Maximum Z: " << zMax << std::endl;

    Vertices filteredPoints;
    filterPointsByZRange(vertices_target, zMin, zMax, filteredPoints);

    std::cout << "target: " << vertices_target.rows() << "x" << vertices_target.cols() << std::endl;
    std::cout << "filteredPoints: " << filteredPoints.rows() << "x" << filteredPoints.cols() << std::endl;

    Eigen::Vector3d point_on_axis = line_direction.head(3); // 对称轴上的点
    Eigen::Vector3d axis_direction = line_direction.tail(3); // 对称轴的方向向量

    // scaling
    Eigen::Vector3d source_scale, target_scale;
    source_scale = vertices_source.rowwise().maxCoeff() - vertices_source.rowwise().minCoeff();
    target_scale = vertices_target.rowwise().maxCoeff() - vertices_target.rowwise().minCoeff();
    double scale = std::max(source_scale.norm(), target_scale.norm());
    std::cout << "scale = " << scale << std::endl;
    vertices_source /= scale;
    vertices_target /= scale;

    /// De-mean
    VectorN source_mean, target_mean;//点云在每个坐标轴上的均值
    source_mean = vertices_source.rowwise().sum() / double(vertices_source.cols());//将每列的元素相加并除以列数得到的。
    target_mean = vertices_target.rowwise().sum() / double(vertices_target.cols());//将每列的元素相加并除以列数得到的。
    vertices_source.colwise() -= source_mean;//将源点云和目标点云的每个坐标都减去对应坐标轴上的均值。
    vertices_target.colwise() -= target_mean;//将源点云和目标点云的每个坐标都减去对应坐标轴上的均值。

    Eigen::Vector3d scaled_axis_direction = axis_direction / scale;
    Eigen::Vector3d transformed_point_on_axis = (point_on_axis - source_mean) / scale + target_mean;

    Eigen::VectorXd transformed_symmetry_axis(6);
    transformed_symmetry_axis << transformed_point_on_axis, scaled_axis_direction;  

    double time;

    //读取控制点
    std::string filename_controlPCD = "./data/0325_dots.txt"; // 修改为实际文件名
    std::string filename_s1 = out_path + "s1_tie.txt"; // 修改为实际文件名
    std::string filename_s2 = out_path + "s2_tie.txt"; // 修改为实际文件名
    // 从文件中读取点云数据
    PointCloud_control pointCloud_control = readcontrolPointCloudFromFile(filename_controlPCD);
    PointCloud_control pointCloud_control_source = pointCloud_control;
    Vertices p_control = pointCloud_control;
    PointCloud_control pointCloud_s1 = readcontrolPointCloudFromFile(filename_s1);
    PointCloud_control pointCloud_s1_source = pointCloud_s1;
    Vertices pointCloud_s1_tie = pointCloud_s1;
    Vertices p1;
    double radius = 10;

    std::vector<std::vector<double>> p_distance;
    for (int j = 0; j < pointCloud_control.cols(); ++j)
    {
        std::vector<double> p_distance_each;
        for (int i = 0; i < pointCloud_s1_tie.cols(); ++i)
        // for (int j = 0; j < pointCloud_control.cols(); ++j)
        {
            Eigen::Vector3d point_s1 = pointCloud_s1_tie.col(i);
            Eigen::Vector3d point_control = pointCloud_control.col(j);
            double dis = (point_control - point_s1).norm();
            p_distance_each.push_back(dis);
        }
        p_distance.push_back(p_distance_each);
    }

    std::cout << pointCloud_s1_tie << std::endl;
    std::cout << "\nsize of p_distance :" << p_distance.size() << std::endl;
    std::cout << "size of p1_tie :" << pointCloud_s1_tie.cols() << std::endl;
    
    findNeighborsWithinRadius(pointCloud_s1_tie,vertices_target_input,radius,p1);
    std::cout << "size of p1 :" << p1.cols() << std::endl;
    PointCloud_control pointCloud_s2 = readcontrolPointCloudFromFile(filename_s2);
    PointCloud_control pointCloud_s2_source = pointCloud_s2;
    Vertices pointCloud_s2_tie = pointCloud_s2;
    
    std::vector<Eigen::Vector3d> p2_neighbor;
    std::vector<Vertices> p2_neighbor_all;
    std::vector<Vertices> p1_neighbor_all;

    for (int i = 0; i < pointCloud_s1_tie.cols(); ++i)
    {
        Eigen::Vector3d point_s1 = pointCloud_s1_tie.col(i);
      
        Vertices nearestPoints = findNearestPoints(vertices_target_input, point_s1, 2000);
        p1_neighbor_all.push_back(nearestPoints);
    }
   
    Vertices merged_vertices_p1 = mergeVertices(p1_neighbor_all);

    
    std::cout << "Merged Vertices size p1: " << merged_vertices_p1.cols() << std::endl;
    
    for (int i = 0; i < pointCloud_s2_tie.cols(); ++i)
    {
        Eigen::Vector3d point_s2 = pointCloud_s2_tie.col(i);
        // 找到离给定点最近的500个点
        Vertices nearestPoints = findNearestPoints(vertices_source_input, point_s2, 20);
        p2_neighbor_all.push_back(nearestPoints);

        std::vector<std::vector<double>> p_distance_2;
        for(int j = 0; j < pointCloud_control.cols(); j++)
        {
            std::vector<double> p_distance_each_2;
            Eigen::Vector3d point_c = pointCloud_control.col(j);
            for(int k = 0; k < nearestPoints.cols(); k++)
            {
                Eigen::Vector3d point_n = nearestPoints.col(k);
                double n = (point_c - point_n).norm();
                p_distance_each_2.push_back(n);
            }
            p_distance_2.push_back(p_distance_each_2);
        }

        int min_index;
        for(int f = 0; f < pointCloud_control.cols(); f++)
        {
            std::vector<double> p_dis;
            for(int j = 0;j < p_distance_2[f].size(); j++)
            {
                double dl = std::abs(p_distance[f][i] - p_distance_2[f][j]);
                p_dis.push_back(dl);
            }
            // 找到 p_dis 中的最小值的迭代器
            auto min_it = std::min_element(p_dis.begin(), p_dis.end());

            // 计算最小值在 p_dis 中的索引
            min_index = std::distance(p_dis.begin(), min_it);
        }
        // std::cout << "int min :" << min_index << std::endl;
        Eigen::Vector3d p2_p = nearestPoints.col(min_index);
        // std::cout << "int min :" << std::endl;
        p2_neighbor.push_back(p2_p);
    }
    
    Vertices merged_vertices_p2 = mergeVertices(p2_neighbor_all);


    std::cout << "Merged Vertices size: p2 " << merged_vertices_p2.cols() << std::endl;

    std::cout << "size of p2_nearests :" << p2_neighbor.size() << std::endl;
    std::cout << "size of p2_nearests_all :" << p2_neighbor_all.size() << std::endl;
    Vertices p2(3,p2_neighbor.size());
    for(int i = 0;i < p2_neighbor.size();i++)
    {
        p2.col(i) = p2_neighbor[i];
    }
    // std::cout <<  p2 << std::endl;

    // scaling
    Eigen::Vector3d source_scale_p2, target_scale_p1;
    source_scale_p2 = merged_vertices_p2.rowwise().maxCoeff() - merged_vertices_p2.rowwise().minCoeff();
    target_scale_p1 = merged_vertices_p1.rowwise().maxCoeff() - merged_vertices_p1.rowwise().minCoeff();
    double scale_cut = std::max(source_scale_p2.norm(), target_scale_p1.norm());
    std::cout << "scale = " << scale_cut << std::endl;
    merged_vertices_p2 /= scale_cut;
    merged_vertices_p1 /= scale_cut;

    /// De-mean
    VectorN source_mean_p2, target_mean_p1;//点云在每个坐标轴上的均值
    source_mean_p2 = merged_vertices_p2.rowwise().sum() / double(merged_vertices_p2.cols());//将每列的元素相加并除以列数得到的。
    target_mean_p1 = merged_vertices_p1.rowwise().sum() / double(merged_vertices_p1.cols());//将每列的元素相加并除以列数得到的。
    merged_vertices_p2.colwise() -= source_mean_p2;//将源点云和目标点云的每个坐标都减去对应坐标轴上的均值。
    merged_vertices_p1.colwise() -= target_mean_p1;//将源点云和目标点云的每个坐标都减去对应坐标轴上的均值。

    // std::cout << "size of vertices_source_input :" << vertices_source_input.cols() << std::endl;
    // findNeighborsWithinRadius(pointCloud_s2_tie,vertices_source_input,radius,p2);
    // std::cout << "p2_tie :" << p2 << "\n" << std::endl;


 
    if (pointCloud_control.cols() > 0 && pointCloud_s1.cols() > 0 && pointCloud_s2.cols() > 0) {
        std::cout << "\n\nSuccessfully read " << pointCloud_control.cols() << " contorl points from file." << std::endl;
        std::cout << "Successfully read " << pointCloud_s1.cols() << " s1 points from file." << std::endl;
        std::cout << "Successfully read " << pointCloud_s2.cols() << " s2 points from file.\n" << std::endl;
    } else {
        std::cout << "Failed to read point cloud data from file." << std::endl;
    }
   
    scaleAndDeMeanPointCloud(pointCloud_control);
    scaleAndDeMeanPointCloud(pointCloud_s1);
    scaleAndDeMeanPointCloud(pointCloud_s2);

 
    // std::cout << "Scaled and de-meaned point cloud:\n" << pointCloud_s1 << std::endl;

    std::vector<std::vector<double>> distance_pointcloud_control;
    std::vector<double> distances_each;
    for (int i = 0; i < pointCloud_control.cols(); ++i) {
        VectorN point_control = pointCloud_control.col(i);
        for(int j = 0; j < pointCloud_s1.cols();j++)
        {
            VectorN Point = pointCloud_s1.col(j);
            double distance = calculateDistance(point_control, Point);
            distances_each.push_back(distance);
        }

        distance_pointcloud_control.push_back(distances_each);  
    }

    // std::cout << "\ndistance all " << distance_pointcloud_control.size() << " points\n" << std::endl;

    std::vector<std::vector<double>> distance_pointcloud_control_source;
    std::vector<double> distances_each_source;
    for (int i = 0; i < pointCloud_control_source.cols(); ++i) {
        VectorN point_control_source = pointCloud_control_source.col(i);
        for(int j = 0; j < pointCloud_s1_source.cols();j++)
        {
            VectorN Point = pointCloud_s1_source.col(j);
            double distance_source = calculateDistance(point_control_source, Point);
            distances_each_source.push_back(distance_source);
        }

        distance_pointcloud_control_source.push_back(distances_each_source);  
    }


    std::cout << "\ndistance all " << distance_pointcloud_control.size() << " points\n" << std::endl;
    // set ICP parameters
    ICP::Parameters pars;

    // set Sparse-ICP parameters
    SICP::Parameters spars;
    spars.p = 0.4;
    spars.print_icpn = false;

    Matrix4x4 init_trans_4_4;
    /// Initial transformation
    if(use_init)
    {
        MatrixXX init_trans;
        file_init = out_path + "initial_matrix.txt";
        read_transMat(init_trans, file_init);
        init_trans_4_4 = init_trans;
        init_trans.block(0, dim, dim, 1) /= scale;
        init_trans.block(0,3,3,1) += init_trans.block(0,0,3,3)*source_mean - target_mean;
        pars.use_init = true;
        pars.init_trans = init_trans;
        spars.init_trans = init_trans;

       
        Eigen::Matrix3d rotation_matrix_angle = init_trans.block<3, 3>(0, 0);


       
        Eigen::Vector3d euler_angles_init = rotation_matrix_angle.eulerAngles(2, 1, 0); // ZYX顺序


    }

    
    PointCloud_control transformed_s2_source = transformPointCloud(pointCloud_s2_source, init_trans_4_4);
    PointCloud_control transformed_s2 = transformPointCloud(pointCloud_s2, init_trans_4_4);
    std::vector<std::vector<double>> distance_pointcloud_control_source_s2;
    std::vector<double> distances_each_source_s2;
    for (int i = 0; i < pointCloud_control_source.cols(); ++i) {
        VectorN point_control_source_s2 = pointCloud_control_source.col(i);
        for(int j = 0; j < transformed_s2_source.cols();j++)
        {
            VectorN Point = transformed_s2_source.col(j);
            double distance_source_s2 = calculateDistance(point_control_source_s2, Point);
            distances_each_source_s2.push_back(distance_source_s2);
        }

        distance_pointcloud_control_source_s2.push_back(distances_each_source_s2);  
    }
    int count = pointCloud_s2_tie.cols() * pointCloud_control_source.cols();
    double sum = 0;
    double sumSquaredError1 = 0;
    for(int i = 0;i < distance_pointcloud_control_source_s2.size(); i++)
    {
        for( int j = 0; j < distance_pointcloud_control_source_s2[0].size();j++)
        {
            sum += std::abs(distance_pointcloud_control_source_s2[i][j] - distance_pointcloud_control_source[i][j]) * 2;
            double diff = std::abs(distance_pointcloud_control_source_s2[i][j] - distance_pointcloud_control_source[i][j]);
            sumSquaredError1 += diff * diff * 2;
           
        }
    }
    double rmse1 = std::sqrt(sumSquaredError1 / count);
    double mean_source = sum / count;
    
    
    ///--- Execute registration
    std::cout << "begin registration..." << std::endl;
    FRICP<3> fricp;
    double begin_reg = omp_get_wtime();
    double converge_rmse = 0;
    switch(method)
    {
    case ICP:
    {
        pars.f = ICP::NONE;
        pars.use_AA = false;
        // fricp.point_to_point(merged_vertices_p2, merged_vertices_p1, source_mean_p2, target_mean_p1, pars,
        //     line_direction,distance_pointcloud_control,pointCloud_s1,pointCloud_s2,pointCloud_control,pointCloud_s1_source,
        //     pointCloud_s2_source,pointCloud_control_source);
        fricp.point_to_point(merged_vertices_p2, merged_vertices_p1, source_mean_p2, target_mean_p1, pars,
            line_direction,distance_pointcloud_control,pointCloud_s1,pointCloud_s2,pointCloud_control,pointCloud_s1_source,
            pointCloud_s2_source,pointCloud_control_source);
        res_trans = pars.res_trans;
        break;
    }
    case AA_ICP:
    {
        AAICP::point_to_point_aaicp(vertices_source, vertices_target, source_mean, target_mean, pars);
        res_trans = pars.res_trans;
        break;
    }
    case FICP:
    {
        pars.f = ICP::NONE;
        fricp.point_to_point(vertices_source, vertices_target, source_mean_p2, target_mean_p1, pars,
            line_direction,distance_pointcloud_control,pointCloud_s1,pointCloud_s2,pointCloud_control,pointCloud_s1_source,
            pointCloud_s2_source,pointCloud_control_source);
        res_trans = pars.res_trans;
        break;
    }
    case RICP:
    {
        pars.f = ICP::WELSCH;
        pars.use_AA = true;
        fricp.point_to_point(vertices_source, vertices_target, source_mean, target_mean, pars,
            line_direction,distance_pointcloud_control,pointCloud_s1,pointCloud_s2,pointCloud_control,pointCloud_s1_source,
            transformed_s2_source,pointCloud_control_source);
            // fricp.point_to_point(vertices_source, vertices_target, source_mean_p2, target_mean_p1, pars,
            // line_direction,distance_pointcloud_control,pointCloud_s1,pointCloud_s2,pointCloud_control);
        res_trans = pars.res_trans;
        break;
    }
    case PPL:
    {
        pars.f = ICP::NONE;
        pars.use_AA = false;
        if(normal_target.size()==0)
        {
            std::cout << "Warning! The target model without normals can't run Point-to-plane method!" << std::endl;
            exit(0);
        }
        fricp.point_to_plane(vertices_source, vertices_target, normal_source, normal_target, source_mean, target_mean, pars);
        res_trans = pars.res_trans;
        break;
    }
    case RPPL:
    {
        pars.nu_end_k = 1.0/6;
        pars.f = ICP::WELSCH;
        pars.use_AA = true;
        if(normal_target.size()==0)
        {
            std::cout << "Warning! The target model without normals can't run Point-to-plane method!" << std::endl;
            exit(0);
        }
        fricp.point_to_plane_GN(vertices_source, vertices_target, normal_source, normal_target, source_mean, target_mean, pars);
        res_trans = pars.res_trans;
        break;
    }
    case SparseICP:
    {
        SICP::point_to_point(vertices_source, vertices_target, source_mean, target_mean, spars);
        res_trans = spars.res_trans;
        break;
    }
    case SICPPPL:
    {
        if(normal_target.size()==0)
        {
            std::cout << "Warning! The target model without normals can't run Point-to-plane method!" << std::endl;
            exit(0);
        }
        SICP::point_to_plane(vertices_source, vertices_target, normal_target, source_mean, target_mean, spars);
        res_trans = spars.res_trans;
        break;
    }
    }
	std::cout << "Registration done!" << "\n" << std::endl;
    double end_reg = omp_get_wtime();
    time = end_reg - begin_reg;
    vertices_source = scale * vertices_source;
    merged_vertices_p2 = scale_cut * merged_vertices_p2;
    // vertices_target = scale * vertices_target;

    out_path = out_path + "m" + std::to_string(method);
    Eigen::Affine3d res_T;
    res_trans.block(0,3,3,1) = init_trans_4_4.block(0,3,3,1);
    res_trans.block(0,0,3,3) = init_trans_4_4.block(0,0,3,3);
    res_T.linear() = res_trans.block(0,0,3,3);
    res_T.translation() = res_trans.block(0,3,3,1);
    res_trans_path = out_path + "trans.txt";
    

    std::ofstream out_trans(res_trans_path);
    // res_trans.block(0,3,3,1) *= scale_cut;
    // std::cout << "res_trans :\n" << res_trans << std::endl;
    Eigen::Matrix4f transformation_T_ICP;
    transformation_T_ICP.block<4, 4>(0, 0) = res_trans.cast<float>();
    
    Vertices transformed_vertices(vertices_source_input.rows(), vertices_source_input.cols());

    for (int i = 0; i < vertices_source_input.cols(); ++i) {
        Eigen::Vector4d point_homogeneous;
        point_homogeneous.head(3) = vertices_source_input.col(i);
        point_homogeneous(3) = 1.0; // 设置齐次坐标的最后一个分量为1

        Eigen::Vector4d transformed_point = res_trans * point_homogeneous; // 应用转换矩阵
        transformed_vertices.col(i) = transformed_point.head(3); // 存储变换后的点
    }
    Eigen::Matrix4f transformation_T = Eigen::Matrix4f::Identity();


    Vertices aligned_pointcloud = ICP_pcl(transformed_vertices,vertices_target_input,transformation_T);
    transformation_T = transformation_T * transformation_T_ICP;

    size_t N = filteredPoints.cols();  


    double distanceThreshold = 0.1;  

 
    Eigen::MatrixXi indices;
    Eigen::MatrixXd distances;

    Vertices Q;
    find_nearest_neighbors(vertices_target_input, transformed_vertices, indices, distances, distanceThreshold,Q,K);

    // std::cout << "Q: " << Q.rows() << "x" << Q.cols() << std::endl;
    // std::cout << "K: " << K.rows() << "x" << K.cols() << std::endl;

 
    if (Q.cols() != K.cols()) {
        std::cerr << "Error: Sizes of K and Q do not match." << std::endl;
        return -1;
    }

    VectorX L1 = VectorX::Zero(Q.cols());
    VectorX L2 = VectorX::Zero(Q.cols());
    VectorX L = VectorX::Zero(Q.cols());
    int nPoints = Q.cols();
    

    // Q = transformation_matrix * Q;
    // vertices_source = transformation_matrix * vertices_source;
    for (int i = 0; i<nPoints; ++i) {
    Eigen::VectorXd pointX = Q.col(i);
    Eigen::VectorXd pointL = line_direction.head(3);
    Eigen::VectorXd pointQ = K.col(i);
    Eigen::VectorXd line_direction_normal = line_direction.tail(3);

    double perpendicular_distance1 = point_to_line_perpendicular_distance(pointX, pointL, line_direction_normal);
    double perpendicular_distance2 = point_to_line_perpendicular_distance(pointQ, pointL, line_direction_normal);

    // std::cout << "perpendicular_distance1" << perpendicular_distance1 << "\n" << std::endl;
    // std::cout << "perpendicular_distance2" << perpendicular_distance2 << "\n" << std::endl;

    L1[i] = perpendicular_distance1;
    L2[i] = perpendicular_distance2;
    L[i] = std::abs(perpendicular_distance1 - perpendicular_distance2);
    }

    std::cout << "Transformation matrix saved to transformation.txt" << std::endl;
    double max_value = L.maxCoeff();
   

    double mean = L.mean();
   

    // Specify the filename
    std::string filename = out_path + "_vertices_source.txt";

    std::string filename_L = out_path + "_distance.txt";

    // Write vertices_source to the file
    writeVerticesToFile(transformed_vertices, filename);

 
    out_trans << transformation_T << std::endl;

  
    out_trans.close();


    
    double rmse = calculate_rmse(Q, K);


    Matrix4x4 transformation_T_converted = transformation_T.cast<Scalar>();

    PointCloud_control transformed_s2_final = transformPointCloud(pointCloud_s2_source, transformation_T_converted);
    std::vector<std::vector<double>> distance_pointcloud_control_final_s2;
    std::vector<double> distances_each_final_s2;
    for (int i = 0; i < pointCloud_control_source.cols(); ++i) {
        VectorN point_control_source_s2 = pointCloud_control_source.col(i);
        for(int j = 0; j < transformed_s2_final.cols();j++)
        {
            VectorN Point = transformed_s2_final.col(j);
            double distance_source_s2_final = calculateDistance(point_control_source_s2, Point);
            distances_each_final_s2.push_back(distance_source_s2_final);
        }

        distance_pointcloud_control_final_s2.push_back(distances_each_final_s2);  
    }

    double sum_f = 0;
    double sumSquaredError = 0.0;
    for(int i = 0;i < distance_pointcloud_control_final_s2.size(); i++)
    {
        for( int j = 0; j < distance_pointcloud_control_final_s2[0].size();j++)
        {
            sum_f += std::abs(distance_pointcloud_control_final_s2[i][j] - distance_pointcloud_control_source[i][j]);
            double diff = std::abs(distance_pointcloud_control_final_s2[i][j] - distance_pointcloud_control_source[i][j]);
            sumSquaredError += diff * diff;

        }
    }
    double rmse2 = std::sqrt(sumSquaredError / count);

    std::cout << "RMSE: " << rmse << std::endl;
    double mean_final = sum_f / count;
    std::cout << "\ncount :" << count <<  std::endl;
    std::cout << "\n配准结果 :" <<  std::endl;
    std::cout << "RMSE: " << rmse << std::endl;
    std::cout << "到对称轴线的最大距离 :" << max_value << std::endl;
    std::cout << "到对称轴线的平均距离 :" << mean << "\n" <<std::endl;
    
    std::cout << "标准点云到控制点距离 : " <<  mean_source << std::endl;
    std::cout << "配准点云到控制点距离 : " <<  mean_final << std::endl;
    std::cout << "配准点云到控制点距离rmse1 : " <<  rmse1 << std::endl;
    std::cout << "配准点云到控制点距离rmse2 : " <<  rmse2 << std::endl;



    // // Specify the filename
    // std::string filename_ceres = "/home/wcz/fast_icp/src/Fast-Robust-ICP-master/data/matrix/ceres_vertices.txt";

    // // Write vertices_source to the file

    // writeVerticesToFile(vertices_merge_cloud, filename_merge);


    ///--- Write result to file
    // std::string file_source_reg = out_path + "reg_pc.ply";
    // write_file(file_source, vertices_source, normal_source, src_vert_colors, file_source_reg);

    std::cout << "Press any key to exit..." << std::endl;
    std::cin.get();

    return 0;
}
