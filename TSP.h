#ifndef FRICP_H
#define FRICP_H
#include "ICP.h"
#include <AndersonAcceleration.h>
#include <unsupported/Eigen/MatrixFunctions>
#include "median.h"
#include <limits>
#define SAME_THRESHOLD 1e-6
#include <type_traits>
#include <Eigen/Dense>
#include <Eigen/SVD>
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <ceres/ceres.h>
#include <numeric>
#include <flann/flann.hpp>
#include <algorithm>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> PointCloud_control;//读取控制点
typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> PointCloud;

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
            // unless the result is subnormal
            || std::fabs(x-y) < std::numeric_limits<T>::min();
}
template<int N>
class FRICP
{
public:
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, N, Eigen::Dynamic> MatrixNX;
    typedef Eigen::Matrix<Scalar, N, N> MatrixNN;
    typedef Eigen::Matrix<Scalar, N+1, N+1> AffineMatrixN;
    typedef Eigen::Transform<Scalar, N, Eigen::Affine> AffineNd;
    typedef Eigen::Matrix<Scalar, N, 1> VectorN;
    typedef nanoflann::KDTreeAdaptor<MatrixNX, N, nanoflann::metric_L2_Simple> KDtree;
    typedef Eigen::Matrix<Scalar, 6, 1> Vector6;
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Vertices;//Eigen矩阵类型，其大小是3xN，表示点云的顶点坐标。
    double test_total_construct_time=.0;
    double test_total_solve_time=.0;
    int test_total_iters=0;

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

    FRICP(){};
    ~FRICP(){};

private:
    AffineMatrixN LogMatrix(const AffineMatrixN& T)
    {
        Eigen::RealSchur<AffineMatrixN> schur(T);
        AffineMatrixN U = schur.matrixU();
        AffineMatrixN R = schur.matrixT();
        std::vector<bool> selected(N, true);
        MatrixNN mat_B = MatrixNN::Zero(N, N);
        MatrixNN mat_V = MatrixNN::Identity(N, N);

        for (int i = 0; i < N; i++)
        {
            if (selected[i] && fabs(R(i, i) - 1)> SAME_THRESHOLD)
            {
                int pair_second = -1;
                for (int j = i + 1; j <N; j++)
                {
                    if (fabs(R(j, j) - R(i, i)) < SAME_THRESHOLD)
                    {
                        pair_second = j;
                        selected[j] = false;
                        break;
                    }
                }
                if (pair_second > 0)
                {
                    selected[i] = false;
                    R(i, i) = R(i, i) < -1 ? -1 : R(i, i);
                    double theta = acos(R(i, i));
                    if (R(i, pair_second) < 0)
                    {
                        theta = -theta;
                    }
                    mat_B(i, pair_second) += theta;
                    mat_B(pair_second, i) += -theta;
                    mat_V(i, pair_second) += -theta / 2;
                    mat_V(pair_second, i) += theta / 2;
                    double coeff = 1 - (theta * R(i, pair_second)) / (2 * (1 - R(i, i)));
                    mat_V(i, i) += -coeff;
                    mat_V(pair_second, pair_second) += -coeff;
                }
            }
        }

        AffineMatrixN LogTrim = AffineMatrixN::Zero();
        LogTrim.block(0, 0, N, N) = mat_B;
        LogTrim.block(0, N, N, 1) = mat_V * R.block(0, N, N, 1);
        AffineMatrixN res = U * LogTrim * U.transpose();
        return res;
    }

    inline Vector6 RotToEuler(const AffineNd& T)
    {
        Vector6 res;
        res.head(3) = T.rotation().eulerAngles(0,1,2);
        res.tail(3) = T.translation();
        return res;
    }

    inline AffineMatrixN EulerToRot(const Vector6& v)
    {
        MatrixNN s (Eigen::AngleAxis<Scalar>(v(0), Vector3::UnitX())
                    * Eigen::AngleAxis<Scalar>(v(1), Vector3::UnitY())
                    * Eigen::AngleAxis<Scalar>(v(2), Vector3::UnitZ()));

        AffineMatrixN m = AffineMatrixN::Zero();
        m.block(0,0,3,3) = s;
        m(3,3) = 1;
        m.col(3).head(3) = v.tail(3);
        return m;
    }
    inline Vector6 LogToVec(const Eigen::Matrix4d& LogT)
    {
        Vector6 res;
        res[0] = -LogT(1, 2);
        res[1] = LogT(0, 2);
        res[2] = -LogT(0, 1);
        res[3] = LogT(0, 3);
        res[4] = LogT(1, 3);
        res[5] = LogT(2, 3);
        return res;
    }

    inline AffineMatrixN VecToLog(const Vector6& v)
    {
        AffineMatrixN m = AffineMatrixN::Zero();
        m << 0, -v[2], v[1], v[3],
                v[2], 0, -v[0], v[4],
                -v[1], v[0], 0, v[5],
                0, 0, 0, 0;
        return m;
    }

    double FindKnearestMed(const KDtree& kdtree,
                           const MatrixNX& X, int nk)
    {
        Eigen::VectorXd X_nearest(X.cols());
#pragma omp parallel for
        for(int i = 0; i<X.cols(); i++)
        {
            int* id = new int[nk];
            double *dist = new double[nk];
            kdtree.query(X.col(i).data(), nk, id, dist);
            Eigen::VectorXd k_dist = Eigen::Map<Eigen::VectorXd>(dist, nk);
            igl::median(k_dist.tail(nk-1), X_nearest[i]);
            delete[]id;
            delete[]dist;
        }
        double med;
        igl::median(X_nearest, med);
        return sqrt(med);
    }
    /// Find self normal edge median of point cloud
    double FindKnearestNormMed(const KDtree& kdtree, const Eigen::Matrix3Xd & X, int nk, const Eigen::Matrix3Xd & norm_x)
    {
        Eigen::VectorXd X_nearest(X.cols());
#pragma omp parallel for
        for(int i = 0; i<X.cols(); i++)
        {
            int* id = new int[nk];
            double *dist = new double[nk];
            kdtree.query(X.col(i).data(), nk, id, dist);
            Eigen::VectorXd k_dist = Eigen::Map<Eigen::VectorXd>(dist, nk);
            for(int s = 1; s<nk; s++)
            {
                k_dist[s] = std::abs((X.col(id[s]) - X.col(id[0])).dot(norm_x.col(id[0])));
            }
            igl::median(k_dist.tail(nk-1), X_nearest[i]);
            delete[]id;
            delete[]dist;
        }
        double med;
        igl::median(X_nearest, med);
        return med;
    }



    // struct PointToPointCostFunction {
    // PointToPointCostFunction(const double& sx, const double& sy, const double& sz,
    //                         const double& tx, const double& ty, const double& tz,
    //                         const Eigen::VectorXd& line_direction)
    //     : source_(sx, sy, sz),  // 创建源点的3D向量
    //         target_(tx, ty, tz),  // 创建目标点的3D向量
    //         line_direction_(line_direction.normalized()){}

    //     template <typename T>
    //     bool operator()(const T* const parameters, T* residuals) const {
    //         // Extract parameters
    //         const T& Tx = parameters[0];
    //         const T& Ty = parameters[1];
    //         const T& Tz = parameters[2];
    //         const T* quaternion = parameters + 3;

    //         // Transform source point
    //         Eigen::Matrix<T, 3, 1> sourcePoint;
    //         Eigen::Matrix<T, 3, 1> targetPoint;
    //         Eigen::Matrix<T, 3, 1> targetPoint_trans;

    //         sourcePoint << T(source_(0)), T(source_(1)), T(source_(2));
    //         targetPoint << T(target_(0)), T(target_(1)), T(target_(2));

    //         // Construct transformation matrix
    //         Eigen::Transform<T, 3, Eigen::Affine> transform_matrix;
    //         transform_matrix.translation() << Tx, Ty, Tz;
    //         transform_matrix.linear() = Eigen::Quaternion<T>(quaternion).toRotationMatrix();
    //         // std::cout << "Transformation matrix:\n" << transform_matrix.matrix() << std::endl;
            
    //         targetPoint_trans = transform_matrix * sourcePoint;
    //         // std::cout << "targetPoint_trans: " << targetPoint_trans << std::endl;
    //         // std::cout << "targetPoint: " << targetPoint << std::endl;

    //         Eigen::Matrix<T, 3, 1> line_Point = line_direction_.template cast<T>().head(3);
    //         Eigen::Matrix<T, 3, 1> line_Normal = line_direction_.template cast<T>().tail(3);
    //         Eigen::Matrix<T, 3, 1> sourcePoint3D = sourcePoint.template head<3>().template cast<T>();
    //         Eigen::Matrix<T, 3, 1> targetPoint3D = targetPoint.template head<3>().template cast<T>();
    //         T distance_source = FRICP<N>::pointToLineDistance(targetPoint_trans, line_Point, line_Normal);
    //         T distance_target = FRICP<N>::pointToLineDistance(targetPoint3D, line_Point, line_Normal);

    //         // Compute the residual as the difference in distances
    //         residuals[0] = abs(distance_target - distance_source);
    //         // std::cout << "distance_source: " << distance_source << std::endl;
    //         // std::cout << "distance_target: " << distance_target << std::endl;
    //         // std::cout << "residuals: " << residuals[0] << std::endl;
    //     return true;
    // }

    // private:
    // const Eigen::Vector3d source_;
    // const Eigen::Vector3d target_;
    // Eigen::VectorXd line_direction_;
    // typedef double Scalar;
    // typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> PointCloud;
    // typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
    // };

        struct PointToPointCostFunction {
        PointToPointCostFunction(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const std::vector<std::vector<double>>& distances,const int& i,const int& j)
            : a_(a), b_(b), distances_(distances),i_(i),j_(j){}

        template <typename T>
        bool operator()(const T* const R_ptr, const T* const t_ptr, T* residuals) const {
            Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor>> R(R_ptr);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(t_ptr);

            Eigen::Matrix<T, 3, 1> transformed_a = R * a_.template cast<T>() + t;
            Eigen::Matrix<T, 3, 1> diff = transformed_a - b_.template cast<T>();

            int i = i_;
            int j = j_;
            T residual_sum = T(0.0);
            residual_sum += T(distances_[i][j]) - (transformed_a - b_.template cast<T>()).norm();

            residuals[0] = residual_sum;
            return true;
        }

        private:
            const Eigen::Vector3d a_;
            const Eigen::Vector3d b_;
            const std::vector<std::vector<double>>& distances_;
            const int i_;
            const int j_;
        };

        Eigen::Affine3d findRigidBodyTransformation_ceres(const Vertices& A, const Vertices& B, const Vertices& C) {

        // Calculate distances between points in pointCloud1 and pointCloud3
        std::vector<std::vector<double>> distances(A.cols(), std::vector<double>(C.cols(), 0.0));
        for (int i = 0; i < A.cols(); ++i) {
            for (int j = 0; j < C.cols(); ++j) {
                double distance = (A.col(i) - C.col(j)).norm();
                distances[i][j] = distance;
            }
        }

        // Number of corresponding points
        const int num_points =  C.cols();

        // Construct the problem
        ceres::Problem problem;

        // Create variables for rotation matrix R and translation vector t
        double* R_ptr = new double[9];
        double* t_ptr = new double[3];

        // Initialize R to identity and t to zero
        std::fill(R_ptr, R_ptr + 9, 0.0);
        R_ptr[0] = R_ptr[4] = R_ptr[8] = 1.0;
        std::fill(t_ptr, t_ptr + 3, 0.0);

        // Add rotation and translation variables to the problem
        problem.AddParameterBlock(R_ptr, 9);
        problem.AddParameterBlock(t_ptr, 3);

        // Fix the first row of R to avoid reflection
        problem.SetParameterBlockConstant(&R_ptr[0]);

        for (int j = 0; j < B.cols(); j++)
        {
            for (int i = 0; i < C.cols(); ++i) {
                problem.AddResidualBlock(new ceres::AutoDiffCostFunction<PointToPointCostFunction, 1, 9, 3>(
                    new PointToPointCostFunction(B.col(j), C.col(i), distances, j, i)),
                    nullptr, R_ptr, t_ptr);
            }
        }

        // Solve the problem
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        // options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Extract the optimized rotation matrix and translation vector
        Eigen::Matrix3d R_opt = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R_ptr);
        Eigen::Vector3d t_opt(t_ptr);

        // Construct the transformation matrix
        Eigen::Affine3d transformation;
        transformation.linear() = R_opt;
        transformation.translation() = t_opt;

        // Clean up
        delete[] R_ptr;
        delete[] t_ptr;

        // std::cout << "Transformation matrix:\n" << transformation.matrix() << std::endl;

        return transformation;
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

    // Functor for optimization
    struct TransformationOptimizationFunctor {
        typedef double Scalar;  // Define Scalar type for Eigen::NumericalDiff
        typedef Eigen::VectorXd InputType;
        typedef Eigen::VectorXd ValueType;
        typedef Eigen::MatrixXd JacobianType;

        enum {
            InputsAtCompileTime = Eigen::Dynamic,
            ValuesAtCompileTime = Eigen::Dynamic
        };

        const Eigen::MatrixXd& A;
        const Eigen::MatrixXd& C;
        const std::vector<std::vector<Eigen::Vector3d>>& normals;

        TransformationOptimizationFunctor(const Eigen::MatrixXd& A_, const Eigen::MatrixXd& C_, const std::vector<std::vector<Eigen::Vector3d>>& normals_)
            : A(A_), C(C_), normals(normals_) {}

        int inputs() const { return 6; }
        int values() const { return A.cols() * C.cols(); }

        int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const {
            Eigen::Vector3d r_vec = x.head<3>();
            Eigen::Vector3d t = x.tail<3>();

            double theta = r_vec.norm();
            Eigen::Matrix3d R = Eigen::AngleAxisd(theta, r_vec.normalized()).toRotationMatrix();

            int k = 0;
            for (int i = 0; i < A.cols(); ++i) {
                for (int j = 0; j < C.cols(); ++j) {
                    Eigen::Vector3d A_prime = R * A.col(i) + t;
                    Eigen::Vector3d normal = C.col(j) - A_prime;
                    fvec(k++) = (normal - normals[i][j]).norm();
                }
            }
            return 0;
        }
    };

    // Function to compute normals between points in B and C
    std::vector<std::vector<Eigen::Vector3d>> computeNormals(const Eigen::MatrixXd& B, const Eigen::MatrixXd& C) {
        std::vector<std::vector<Eigen::Vector3d>> normals(B.cols(), std::vector<Eigen::Vector3d>(C.cols()));

        for (int i = 0; i < B.cols(); ++i) {
            for (int j = 0; j < C.cols(); ++j) {
                Eigen::Vector3d normal = C.col(j) - B.col(i);
                normals[i][j] = normal;
            }
        }

        return normals;
    }

    Eigen::Affine3d findBodyTransformation(const Vertices& A, const Vertices& B, const Vertices& C) {
        std::vector<std::vector<Eigen::Vector3d>> normals = computeNormals(B, C);

        Eigen::VectorXd x(6);
        x.setZero();

        TransformationOptimizationFunctor functor(A, C, normals);
        Eigen::NumericalDiff<TransformationOptimizationFunctor> numDiff(functor);
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<TransformationOptimizationFunctor>, double> lm(numDiff);

        lm.parameters.maxfev = 2000;
        lm.parameters.xtol = 1.0e-10;
        Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);

        if (status <= Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
            std::cerr << "Optimization failed: " << status << std::endl;
        }
        else {
            std::cout << "Optimization succeeded: " << status << std::endl;
        }

        Eigen::Vector3d r_vec = x.head<3>();
        Eigen::Vector3d t = x.tail<3>();
        double theta = r_vec.norm();
        Eigen::Matrix3d R = Eigen::AngleAxisd(theta, r_vec.normalized()).toRotationMatrix();

        Eigen::Affine3d transformation;
        transformation.linear() = R;
        transformation.translation() = t;

        return transformation;
    }



    //寻找源点云 X 到目标点云 Y 的最优刚性变换
    template <typename Derived1, typename Derived2, typename Derived3>
    //X 是经过变换的原始点云，Y 是 相对应的最近邻点，W 存储点与最近邻点的欧式距离
    AffineNd point_to_point(Eigen::MatrixBase<Derived1>& X,
                            Eigen::MatrixBase<Derived2>& Y,
                            const Eigen::MatrixBase<Derived3>& w) {
        int dim = X.rows();// 获取点云的维度，假设每个点的维度相同。
        /// Normalize weight vector
        Eigen::VectorXd w_normalized = w / w.sum();// 对输入的权重向量 w 进行归一化，以确保权重的总和为1。
        /// De-mean
        Eigen::VectorXd X_mean(dim), Y_mean(dim);//创建用于存储 X 和 Y 点云均值的向量。
        for (int i = 0; i<dim; ++i) {//计算源点云 X 和目标点云 Y 在每个维度上的均值。
            X_mean(i) = (X.row(i).array()*w_normalized.transpose().array()).sum();
            Y_mean(i) = (Y.row(i).array()*w_normalized.transpose().array()).sum();
        }
        X.colwise() -= X_mean;//将均值从源点云和目标点云的每一列中减去，实现去中心化。
        Y.colwise() -= Y_mean;//将均值从源点云和目标点云的每一列中减去，实现去中心化。

        /// Compute transformation
        AffineNd transformation;// 创建一个 AffineNd 对象，用于存储变换矩阵。

        MatrixXX sigma = X * w_normalized.asDiagonal() * Y.transpose();// 计算带有权重的协方差矩阵。
        Eigen::JacobiSVD<MatrixXX> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);//对协方差矩阵进行奇异值分解（SVD）
        if (svd.matrixU().determinant()*svd.matrixV().determinant() < 0.0) {
            VectorN S = VectorN::Ones(dim); S(dim-1) = -1.0;
            transformation.linear() = svd.matrixV()*S.asDiagonal()*svd.matrixU().transpose();
        }//确定旋转矩阵的方向，以确保它是合法的旋转矩阵。
        else {
            transformation.linear() = svd.matrixV()*svd.matrixU().transpose();//计算旋转矩阵。
        }
        transformation.translation() = Y_mean - transformation.linear()*X_mean;//计算平移向量。

        /// Re-apply mean
        X.colwise() += X_mean;//将均值重新添加到源点云和目标点云的每一列中。
        Y.colwise() += Y_mean;//将均值重新添加到源点云和目标点云的每一列中。


        return transformation;//返回计算得到的刚性变换。
    }

    template <typename Derived1, typename Derived2, typename Derived3, typename Derived4, typename Derived5>
    Eigen::Affine3d point_to_plane(Eigen::MatrixBase<Derived1>& X,
                                   Eigen::MatrixBase<Derived2>& Y,
                                   const Eigen::MatrixBase<Derived3>& Norm,
                                   const Eigen::MatrixBase<Derived4>& w,
                                   const Eigen::MatrixBase<Derived5>& u) {
        typedef Eigen::Matrix<double, 6, 6> Matrix66;
        typedef Eigen::Matrix<double, 6, 1> Vector6;
        typedef Eigen::Block<Matrix66, 3, 3> Block33;
        /// Normalize weight vector
        Eigen::VectorXd w_normalized = w / w.sum();
        /// De-mean
        Eigen::Vector3d X_mean;
        for (int i = 0; i<3; ++i)
            X_mean(i) = (X.row(i).array()*w_normalized.transpose().array()).sum();
        X.colwise() -= X_mean;
        Y.colwise() -= X_mean;
        /// Prepare LHS and RHS
        Matrix66 LHS = Matrix66::Zero();
        Vector6 RHS = Vector6::Zero();
        Block33 TL = LHS.topLeftCorner<3, 3>();
        Block33 TR = LHS.topRightCorner<3, 3>();
        Block33 BR = LHS.bottomRightCorner<3, 3>();
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(3, X.cols());

#pragma omp parallel
        {
#pragma omp for
            for (int i = 0; i<X.cols(); i++) {
                C.col(i) = X.col(i).cross(Norm.col(i));
            }
#pragma omp sections nowait
            {
#pragma omp section
                for (int i = 0; i<X.cols(); i++) TL.selfadjointView<Eigen::Upper>().rankUpdate(C.col(i), w(i));
#pragma omp section
                for (int i = 0; i<X.cols(); i++) TR += (C.col(i)*Norm.col(i).transpose())*w(i);
#pragma omp section
                for (int i = 0; i<X.cols(); i++) BR.selfadjointView<Eigen::Upper>().rankUpdate(Norm.col(i), w(i));
#pragma omp section
                for (int i = 0; i<C.cols(); i++) {
                    double dist_to_plane = -((X.col(i) - Y.col(i)).dot(Norm.col(i)) - u(i))*w(i);
                    RHS.head<3>() += C.col(i)*dist_to_plane;
                    RHS.tail<3>() += Norm.col(i)*dist_to_plane;
                }
            }
        }
        LHS = LHS.selfadjointView<Eigen::Upper>();
        /// Compute transformation
        Eigen::Affine3d transformation;
        Eigen::LDLT<Matrix66> ldlt(LHS);
        RHS = ldlt.solve(RHS);
        transformation = Eigen::AngleAxisd(RHS(0), Eigen::Vector3d::UnitX()) *
                Eigen::AngleAxisd(RHS(1), Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(RHS(2), Eigen::Vector3d::UnitZ());
        transformation.translation() = RHS.tail<3>();

        /// Apply transformation
        /// Re-apply mean
        X.colwise() += X_mean;
        Y.colwise() += X_mean;
        transformation.translation() += X_mean - transformation.linear()*X_mean;
        /// Return transformation
        return transformation;
    }

    template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
    double point_to_plane_gaussnewton(const Eigen::MatrixBase<Derived1>& X,
                               const Eigen::MatrixBase<Derived2>& Y,
                               const Eigen::MatrixBase<Derived3>& norm_y,
                               const Eigen::MatrixBase<Derived4>& w,
                               Matrix44 Tk,  Vector6& dir) {
        typedef Eigen::Matrix<double, 6, 6> Matrix66;
        typedef Eigen::Matrix<double, 12, 6> Matrix126;
        typedef Eigen::Matrix<double, 9, 3> Matrix93;
        typedef Eigen::Block<Matrix126, 9, 3> Block93;
        typedef Eigen::Block<Matrix126, 3, 3> Block33;
        typedef Eigen::Matrix<double, 12, 1> Vector12;
        typedef Eigen::Matrix<double, 9, 1> Vector9;
        typedef Eigen::Matrix<double, 4, 2> Matrix42;
        /// Normalize weight vector
        Eigen::VectorXd w_normalized = w / w.sum();
        /// Prepare LHS and RHS
        Matrix66 LHS = Matrix66::Zero();
        Vector6 RHS = Vector6::Zero();

        Vector6 log_T = LogToVec(LogMatrix(Tk));
        Matrix33 B = VecToLog(log_T).block(0, 0, 3, 3);
        double a = log_T[0];
        double b = log_T[1];
        double c = log_T[2];
        Matrix33 R = Tk.block(0, 0, 3, 3);
        Vector3 t = Tk.block(0, 3, 3, 1);
        Vector3 u = log_T.tail(3);

        Matrix93 dbdw = Matrix93::Zero();
        dbdw(1, 2) = dbdw(5, 0) = dbdw(6, 1) = -1;
        dbdw(2, 1) = dbdw(3, 2) = dbdw(7, 0) = 1;
        Matrix93 db2dw = Matrix93::Zero();
        db2dw(3, 1) = db2dw(4, 0) = db2dw(6, 2) = db2dw(8, 0) = a;
        db2dw(0, 1) = db2dw(1, 0) = db2dw(7, 2) = db2dw(8, 1) = b;
        db2dw(0, 2) = db2dw(2, 0) = db2dw(4, 2) = db2dw(5, 1) = c;
        db2dw(1, 1) = db2dw(2, 2) = -2 * a;
        db2dw(3, 0) = db2dw(5, 2) = -2 * b;
        db2dw(6, 0) = db2dw(7, 1) = -2 * c;
        double theta = std::sqrt(a*a + b*b + c*c);
        double st = sin(theta), ct = cos(theta);

        Matrix42 coeff = Matrix42::Zero();
        if (theta>SAME_THRESHOLD)
        {
            coeff << st / theta, (1 - ct) / (theta*theta),
                    (theta*ct - st) / (theta*theta*theta), (theta*st - 2 * (1 - ct)) / pow(theta, 4),
                    (1 - ct) / (theta*theta), (theta - st) / pow(theta, 3),
                    (theta*st - 2 * (1 - ct)) / pow(theta, 4), (theta*(1 - ct) - 3 * (theta - st)) / pow(theta, 5);
        }
        else
            coeff(0, 0) = 1;

        Matrix93 tempB3;
        tempB3.block<3, 3>(0, 0) = a*B;
        tempB3.block<3, 3>(3, 0) = b*B;
        tempB3.block<3, 3>(6, 0) = c*B;
        Matrix33 B2 = B*B;
        Matrix93 temp2B3;
        temp2B3.block<3, 3>(0, 0) = a*B2;
        temp2B3.block<3, 3>(3, 0) = b*B2;
        temp2B3.block<3, 3>(6, 0) = c*B2;
        Matrix93 dRdw = coeff(0, 0)*dbdw + coeff(1, 0)*tempB3
                + coeff(2, 0)*db2dw + coeff(3, 0)*temp2B3;
        Vector9 dtdw = coeff(0, 1) * dbdw*u + coeff(1, 1) * tempB3*u
                + coeff(2, 1) * db2dw*u + coeff(3, 1)*temp2B3*u;
        Matrix33 dtdu = Matrix33::Identity() + coeff(2, 0)*B + coeff(2, 1) * B2;

        Eigen::VectorXd rk(X.cols());
        Eigen::MatrixXd Jk(X.cols(), 6);
#pragma omp for
        for (int i = 0; i < X.cols(); i++)
        {
            Vector3 xi = X.col(i);
            Vector3 yi = Y.col(i);
            Vector3 ni = norm_y.col(i);
            double wi = sqrt(w_normalized[i]);

            Matrix33 dedR = wi*ni * xi.transpose();
            Vector3 dedt = wi*ni;

            Vector6 dedx;
            dedx(0) = (dedR.cwiseProduct(dRdw.block(0, 0, 3, 3))).sum()
                    + dedt.dot(dtdw.head<3>());
            dedx(1) = (dedR.cwiseProduct(dRdw.block(3, 0, 3, 3))).sum()
                    + dedt.dot(dtdw.segment<3>(3));
            dedx(2) = (dedR.cwiseProduct(dRdw.block(6, 0, 3, 3))).sum()
                    + dedt.dot(dtdw.tail<3>());
            dedx(3) = dedt.dot(dtdu.col(0));
            dedx(4) = dedt.dot(dtdu.col(1));
            dedx(5) = dedt.dot(dtdu.col(2));

            Jk.row(i) = dedx.transpose();
            rk[i] = wi * ni.dot(R*xi-yi+t);
        }
        LHS = Jk.transpose() * Jk;
        RHS = -Jk.transpose() * rk;
        Eigen::CompleteOrthogonalDecomposition<Matrix66> cod_(LHS);
        dir = cod_.solve(RHS);
        double gTd = -RHS.dot(dir);
        return gTd;
    }


public:
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> PointCloud;
    typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
    VectorX normalizeWeights(const VectorX& L) {
        // 计算 L 中所有值的总和
        double sum = L.sum();

        // 归一化 L 中的值
        VectorX weights = L / sum;

        return weights;
    }

 
    double calculateDistance(const VectorN& p1, const VectorN& p2) {
        return (p1 - p2).norm();
    }

    double calculateMeanDistance(const AffineNd& T, const PointCloud& pointCloud_control, const PointCloud& pointCloud_S2,
                                    const std::vector<std::vector<double>>& distance_pointcloud_control) 
    {

        PointCloud pointCloud_s2 = T * pointCloud_S2;

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

        double distance_control;
        std::vector<double> distances_control_L;
        for(int i = 0;i < distance_pointcloud_control.size();i++)
        {
            for(int j = 0;j < distance_pointcloud_control[i].size();j++)
            {
                distance_control = std::abs(distance_pointcloud_control[i][j] - distance_pointcloud_control_T[i][j]);
                distances_control_L.push_back(distance_control);
            }
        }
        double sum = 0;
        for(int i = 0;i < distances_control_L.size();i++)
        {
            sum += distances_control_L[i];
        }
        double mean = sum/distances_control_L.size();

        return mean;
    }


    //X 是原始点云，Y 是目标点云
    void point_to_point(MatrixNX& X, MatrixNX& Y, VectorN& source_mean,
                        VectorN& target_mean, ICP::Parameters& par,Eigen::VectorXd& line_direction,
                        std::vector<std::vector<double>>& distance_pointcloud_control,PointCloud_control& pointCloud_s1,
                        PointCloud_control& pointCloud_s2,PointCloud_control& pointCloud_control,PointCloud_control& pointCloud_s1_source,
                        PointCloud_control& pointCloud_s2_source,PointCloud_control& pointCloud_control_source){

        KDtree kdtree(Y);//KD 树，它被用于快速查找目标点云 Y 中最近邻的点
        /// Buffers
        MatrixNX Q = MatrixNX::Zero(N, X.cols());//Q 是一个矩阵，用于存储源点云 X 中每个点在目标点云 Y 上的最近邻点的坐标。
        VectorX W = VectorX::Zero(X.cols());//W 是一个向量，用于存储每个点对的权重
        VectorX L1 = VectorX::Zero(X.cols());//存储距离
        VectorX L2 = VectorX::Zero(X.cols());//存储距离        
        VectorX L = VectorX::Zero(X.cols());//存储距离
        Eigen::VectorXd pointL = line_direction.head(3);
        Eigen::VectorXd line_direction_normal = line_direction.tail(3);
        AffineNd T;//T 是一个 AffineNd 类型的对象，用于表示点云的刚性变换。在初始化时，如果 par.use_init 为真，则采用用户提供的初始变换矩阵 par.init_trans；否则，初始化为单位矩阵。
        AffineNd T01;
        // 获取平移部分
        Eigen::VectorXd translation_vector_source = T.translation();
        if (par.use_init) T.matrix() = par.init_trans;//初始矩阵
        else T = AffineNd::Identity();
        MatrixXX To3 = T.matrix();
        MatrixXX To1 = T.matrix();//To1 和 To2 用于存储当前刚性变换 T 的矩阵表示。在后续的迭代中，它们用于检测ICP算法的收敛条件
        MatrixXX To2 = T.matrix();
        int nPoints = X.cols();//nPoints 存储源点云 X 中点的数量。

        //Anderson Acc para
        AndersonAcceleration accelerator_;//加速ICP算法的收敛。Anderson Acceleration是一种数值优化技术，用于更快地达到收敛。
        AffineNd SVD_T = T;//SVD_T 是一个 AffineNd 类型的对象，用于存储当前的刚性变换。在Anderson Acceleration中，需要在每次迭代中更新当前变换。
        double energy = .0, last_energy = std::numeric_limits<double>::max();

        //ground truth point clouds
        MatrixNX X_gt = X;//X_gt 是一个矩阵，用于存储地面点云。在有地面真值信息的情况下，源点云将根据地面真值进行变换。初始为false
        if(par.has_groundtruth)
        {
            VectorN temp_trans = par.gt_trans.col(N).head(N);
            X_gt.colwise() += source_mean;
            X_gt = par.gt_trans.block(0, 0, N, N) * X_gt;
            X_gt.colwise() += temp_trans - target_mean;
        }

        //output para
        std::string file_out = par.out_path;
        std::vector<double> times, energys, gt_mses;//times、energys 和 gt_mses 是用于存储运行时间、能量和地面真值均方误差的容器，将在后续的迭代中记录这些信息。
        double begin_time, end_time, run_time;//begin_time 记录初始化开始的时间，end_time 记录初始化结束的时间，run_time 用于存储初始化的总运行时间。
        double gt_mse = 0.0;//gt_mse 用于存储地面真值均方误差，初始化为0.0。

        // dynamic welsch paras
        double nu1 = 1, nu2 = 1;//nu1 和 nu2 是动态Welsch参数的初始值。
        double nu3 = 0.3;//定义的距离权重阈值
        double begin_init = omp_get_wtime();//begin_init 记录初始化开始的时间，用于计算初始化的运行时间。

        //并行计算最近邻点
        //Find initial closest point
#pragma omp parallel for
        for (int i = 0; i<nPoints; ++i) {
            VectorN cur_p = T * X.col(i);//计算变换后的源点 cur_p
            Q.col(i) = Y.col(kdtree.closest(cur_p.data()));//利用 kdtree 数据结构找到目标点云 Y 中与变换后的源点 cur_p 最近的点，并将其存储在矩阵 Q 的相应列中。
            // W[i] = (cur_p - Q.col(i)).norm();//计算变换后的源点 cur_p 与其在目标点云中最近邻点 Q.col(i) 之间的欧氏距离，将其作为残差存储在向量 W 中

            Eigen::VectorXd pointX = cur_p;
            Eigen::VectorXd pointQ = Y.col(kdtree.closest(cur_p.data()));
            double perpendicular_distance1 = point_to_line_perpendicular_distance(pointX, pointL, line_direction_normal);
            double perpendicular_distance2 = point_to_line_perpendicular_distance(pointQ, pointL, line_direction_normal);
            L1[i] = perpendicular_distance1;
            L2[i] = perpendicular_distance2;
            L[i] = std::abs(perpendicular_distance1 - perpendicular_distance2);
            // double mean_1 = calculateMeanDistance(T,pointCloud_control,pointCloud_s2,distance_pointcloud_control);
            // W[i] = ((cur_p - Q.col(i)).norm() + std::abs(perpendicular_distance1 - perpendicular_distance2) + mean_1);
            W[i] = ((cur_p - Q.col(i)).norm() + std::abs(perpendicular_distance1 - perpendicular_distance2)*2);
            
            // std::cout << i << std::endl;

        }
        if(par.f == ICP::WELSCH)
        {
            //dynamic welsch, calc k-nearest points with itself;
            nu2 = par.nu_end_k * FindKnearestMed(kdtree, Y, 7);
            double med1;//这是向量 W 的中值（median）。
            igl::median(W, med1);
            nu1 = par.nu_begin_k * med1;
            nu1 = nu1>nu2? nu1:nu2;
        }
        double end_init = omp_get_wtime();
        double init_time = end_init - begin_init;

        //AA init
        accelerator_.init(par.anderson_m, (N + 1) * (N + 1), LogMatrix(T.matrix()).data());//Anderson Acceleration 初始化

        begin_time = omp_get_wtime();//记录开始执行 ICP 循环的时间。
        bool stop1 = false;
        bool stop3 = false;
        double mean_min = 0;
        std::vector<double> mean_all;
        double begin_distance = 0;
        while(!stop1)
        {
            /// run ICP
            int icp = 0;
            for (; icp<par.max_icp; ++icp)//最多进行 par.max_icp 次。
            {
                double mean_1 = calculateMeanDistance(T,pointCloud_control,pointCloud_s2,distance_pointcloud_control);
                // std::cout << "icp:" << icp << std::endl;
                bool accept_aa = false;
                energy = get_energy(par.f, W, nu1) + mean_1 * 1000;
                // energy = get_energy(par.f, W, nu1);
                if (par.use_AA)
                {
                    if (energy < last_energy) {
                        last_energy = energy;
                        // std::cout << "energy :" << last_energy << std::endl;
                        accept_aa = true;
                    }
                    else{
                        accelerator_.replace(LogMatrix(SVD_T.matrix()).data());//替换Anderson Acceleration的历史矩阵。
                        //Re-find the closest point 重新找最近点
#pragma omp parallel for
                        for (int i = 0; i<nPoints; ++i) {
                            VectorN cur_p = SVD_T * X.col(i);
                            Q.col(i) = Y.col(kdtree.closest(cur_p.data()));
                            // W[i] = (cur_p - Q.col(i)).norm();

                         

                            Eigen::VectorXd pointX = cur_p;
                            Eigen::VectorXd pointQ = Y.col(kdtree.closest(cur_p.data()));
                            double perpendicular_distance1 = point_to_line_perpendicular_distance(pointX, pointL, line_direction_normal);
                            double perpendicular_distance2 = point_to_line_perpendicular_distance(pointQ, pointL, line_direction_normal);
                            L1[i] = perpendicular_distance1;
                            L2[i] = perpendicular_distance2;
                            L[i] = std::abs(perpendicular_distance1 - perpendicular_distance2);
                            // double mean_2 = calculateMeanDistance(T,pointCloud_control,pointCloud_s2,distance_pointcloud_control);
                            // W[i] = ((cur_p - Q.col(i)).norm() + std::abs(perpendicular_distance1 - perpendicular_distance2));
                            W[i] = ((cur_p - Q.col(i)).norm());
                        }
                        // double mean_2 = calculateMeanDistance(T,pointCloud_control,pointCloud_s2,distance_pointcloud_control);
                        double energy_k = get_energy(par.f, W, nu1);
                        int integer_digits = static_cast<int>(std::log10(std::abs(energy_k))) + 1;
                        last_energy = energy_k;
                        // last_energy = get_energy(par.f, W, nu1);
                        // std::cout << "refind energy :" << last_energy << std::endl;
                    }
                }
                else
                    last_energy = energy;

                end_time = omp_get_wtime();
                run_time = end_time - begin_time;
                if(par.has_groundtruth)
                {
                    gt_mse = (T*X - X_gt).squaredNorm()/nPoints;
                }

                // save results
                energys.push_back(last_energy);
                times.push_back(run_time);
                gt_mses.push_back(gt_mse);
                // par.print_energy = true;
                if (par.print_energy)
                    std::cout << "icp iter = " << icp << ", Energy = " << last_energy
                             << ", time = " << run_time << std::endl;

                robust_weight(par.f, W, nu1);//根据当前的权重函数和权重参数对误差进行加权，用于提高对离群点的鲁棒性。
                // Rotation and translation update
                T = point_to_point(X, Q, W);//通过点对点的方式更新变换矩阵T，优化点云的对齐。
                std::cout << "T:\n" << T.matrix() << std::endl;

                // Eigen::Affine3d transformation = findBodyTransformation(pointCloud_s1_source,pointCloud_s2_source,pointCloud_control_source);
                // std::cout << "Transformation matrix:\n" << transformation.matrix() << std::endl;

                //Anderson Acc
                SVD_T = T;//将当前的变换矩阵T保存到SVD_T中，用于后续Anderson Acceleration的计算。
                if (par.use_AA)
                {
                    AffineMatrixN Trans = (Eigen::Map<const AffineMatrixN>(accelerator_.compute(LogMatrix(T.matrix()).data()).data(), N+1, N+1)).exp();
                    T.linear() = Trans.block(0,0,N,N);
                    T.translation() = Trans.block(0,N,N,1);
                }

                // Find closest point
#pragma omp parallel for//并行计算每个点的最近邻点，以加速计算。
                for (int i = 0; i<nPoints; ++i) {
                    VectorN cur_p = T * X.col(i) ;//将当前变换应用到源点云的每个点。
                    Q.col(i) = Y.col(kdtree.closest(cur_p.data()));//找到目标点云中与变换后的源点云最近的点。
                    // W[i] = (cur_p - Q.col(i)).norm();//计算权重，表示对应点对之间的距离。

                    Eigen::VectorXd pointX = cur_p;
                    Eigen::VectorXd pointQ = Y.col(kdtree.closest(cur_p.data()));
                    double perpendicular_distance1 = point_to_line_perpendicular_distance(pointX, pointL, line_direction_normal);
                    double perpendicular_distance2 = point_to_line_perpendicular_distance(pointQ, pointL, line_direction_normal);
                    L1[i] = perpendicular_distance1;
                    L2[i] = perpendicular_distance2;
                    L[i] = std::abs(perpendicular_distance1 - perpendicular_distance2);
                    // double mean_3 = calculateMeanDistance(T,pointCloud_control,pointCloud_s2,distance_pointcloud_control);
                    // W[i] = ((cur_p - Q.col(i)).norm() + std::abs(perpendicular_distance1 - perpendicular_distance2) + mean_3);
                    W[i] = ((cur_p - Q.col(i)).norm());
                }

                /// Stopping criteria
                double stop2 = (T.matrix() - To2).norm();//计算变换矩阵的变化量，作为停止条件之一。
                To2 = T.matrix();//更新上一次的变换矩阵。
                if(stop2 < par.stop)//如果变换矩阵的变化小于停止阈值，跳出循环。
                {
                    break;
                }
       
            }


            if(par.f!= ICP::WELSCH)
                stop1 = true;
            else//如果使用的是Welsch权重函数，判断当前的Welsch参数是否足够小。如果足够小，则结束迭代。
            {
                stop1 = fabs(nu1 - nu2)<SAME_THRESHOLD? true: false;
                nu1 = nu1*par.nu_alpha > nu2? nu1*par.nu_alpha : nu2;
                if(par.use_AA)
                {
                    accelerator_.reset(LogMatrix(T.matrix()).data());
                    last_energy = std::numeric_limits<double>::max();
                }

            }

        }

        ///calc convergence energy
        // double mean_3 = calculateMeanDistance(T,pointCloud_control,pointCloud_s2,distance_pointcloud_control);
        // last_energy = get_energy(par.f, W, nu1) + mean_3 * 1000;//计算迭代收敛后的能量值。
        last_energy = get_energy(par.f, W, nu1);//计算迭代收敛后的能量值。

        // 将平移部分设置为单位矩阵
        // T.translation() = translation_vector_source;
        X = T * X;//将源点云应用最终的变换矩阵，得到对齐后的点云。
        int numCols = X.cols(); // 获取矩阵 X 的列数
        // 将 X 转换为 PointCloud 类型
        PointCloud pointCloud_X(3, numCols); // 创建一个大小为3xN的 PointCloud 矩阵，用于存储转换后的数据

        // double mean_4 = calculateMeanDistance(T,pointCloud_control,pointCloud_s2,distance_pointcloud_control);
        // std::cout << "final mean distance :" << mean_4 << std::endl;

        gt_mse = (X-X_gt).squaredNorm()/nPoints;//计算对齐后的点云与地面真值的均方误差。
        T.translation() += - T.rotation() * source_mean + target_mean;//调整平移部分，使得变换后的点云均值与目标点云均值一致。
        double stop4 = (T.matrix() - To2).norm();
       
        X.colwise() += target_mean;

        ///save convergence result
        par.convergence_energy = last_energy;
        par.convergence_gt_mse = gt_mse;
        par.res_trans = T.matrix();

        ///output
        if (par.print_output)
        {
            std::ofstream out_res(par.out_path);
            if (!out_res.is_open())
            {
                std::cout << "Can't open out file " << par.out_path << std::endl;
            }

            //output time and energy
            out_res.precision(16);
            for (int i = 0; i<times.size(); i++)
            {
                out_res << times[i] << " "<< energys[i] << " " << gt_mses[i] << std::endl;
            }
            out_res.close();
            std::cout << " write res to " << par.out_path << std::endl;
        }
    }


    /// Reweighted ICP with point to plane
    /// @param Source (one 3D point per column)
    /// @param Target (one 3D point per column)
    /// @param Target normals (one 3D normal per column)
    /// @param Parameters
    //    template <typename Derived1, typename Derived2, typename Derived3>
    void point_to_plane(Eigen::Matrix3Xd& X,
                        Eigen::Matrix3Xd& Y, Eigen::Matrix3Xd& norm_x, Eigen::Matrix3Xd& norm_y,
                        Eigen::Vector3d& source_mean, Eigen::Vector3d& target_mean,
                        ICP::Parameters &par) {
        /// Build kd-tree
        KDtree kdtree(Y);//构建目标点云 Y 的 KD 树，用于快速查找源点云 X 中每个点的最近邻点。
        /// Buffers

        //定义两个矩阵 Qp 和 Qn，分别用于存储最近邻点的坐标和法向量。这两个矩阵的列数与源点云 X 的点数相同。
        Eigen::Matrix3Xd Qp = Eigen::Matrix3Xd::Zero(3, X.cols());
        Eigen::Matrix3Xd Qn = Eigen::Matrix3Xd::Zero(3, X.cols());
        Eigen::VectorXd W = Eigen::VectorXd::Zero(X.cols());//定义一个向量 W，用于存储每个点对的权重。
        Eigen::Matrix3Xd ori_X = X;//将原始的源点云 X 保存到 ori_X 中，以备后续比较。
        AffineNd T;//定义一个 AffineNd 类型的对象 T，表示刚性变换矩阵。AffineNd 是一个封装了刚性变换的类，包括旋转和平移。
        if (par.use_init) T.matrix() = par.init_trans;
        else T = AffineNd::Identity();
        AffineMatrixN To1 = T.matrix();//将初始变换矩阵 T 保存到 To1 中。
        X = T*X;//将源点云 X 应用初始变换矩阵 T，即进行初始对齐。

        Eigen::Matrix3Xd X_gt = X;
        if(par.has_groundtruth)
        {
            Eigen::Vector3d temp_trans = par.gt_trans.block(0, 3, 3, 1);
            X_gt = ori_X;
            X_gt.colwise() += source_mean;
            X_gt = par.gt_trans.block(0, 0, 3, 3) * X_gt;
            X_gt.colwise() += temp_trans - target_mean;
        }

        std::vector<double> times, energys, gt_mses;
        double begin_time, end_time, run_time;
        double gt_mse = 0.0;

        ///dynamic welsch, calc k-nearest points with itself;
        double begin_init = omp_get_wtime();

        //Anderson Acc para
        AndersonAcceleration accelerator_;
        AffineNd LG_T = T;
        double energy = 0.0, prev_res = std::numeric_limits<double>::max(), res = 0.0;


        // Find closest point
#pragma omp parallel for
        for (int i = 0; i<X.cols(); ++i) {
            int id = kdtree.closest(X.col(i).data());
            Qp.col(i) = Y.col(id);
            Qn.col(i) = norm_y.col(id);
            W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
        }
        double end_init = omp_get_wtime();
        double init_time = end_init - begin_init;

        begin_time = omp_get_wtime();
        int total_iter = 0;
        double test_total_time = 0.0;
        bool stop1 = false;
        while(!stop1)
        {
            /// ICP
            for(int icp=0; icp<par.max_icp; ++icp) {
                total_iter++;

                bool accept_aa = false;
                energy = get_energy(par.f, W, par.p);
                end_time = omp_get_wtime();
                run_time = end_time - begin_time;
                energys.push_back(energy);
                times.push_back(run_time);
                Eigen::VectorXd test_w = (X-Qp).colwise().norm();
                if(par.has_groundtruth)
                {
                    gt_mse = (X - X_gt).squaredNorm()/X.cols();
                }
                gt_mses.push_back(gt_mse);

                /// Compute weights
                robust_weight(par.f, W, par.p);
                /// Rotation and translation update
                T = point_to_plane(X, Qp, Qn, W, Eigen::VectorXd::Zero(X.cols()))*T;
                /// Find closest point
#pragma omp parallel for
                for(int i=0; i<X.cols(); i++) {
                    X.col(i) = T * ori_X.col(i);
                    int id = kdtree.closest(X.col(i).data());
                    Qp.col(i) = Y.col(id);
                    Qn.col(i) = norm_y.col(id);
                    W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
                }

                if(par.print_energy)
                    std::cout << "icp iter = " << total_iter << ", gt_mse = " << gt_mse
                              << ", energy = " << energy << std::endl;

                /// Stopping criteria
                double stop2 = (T.matrix() - To1).norm();
                To1 = T.matrix();
                if(stop2 < par.stop) break;
            }
            stop1 = true;
        }

        par.res_trans = T.matrix();

        ///calc convergence energy
        W = (Qn.array()*(X - Qp).array()).colwise().sum().abs().transpose();
        energy = get_energy(par.f, W, par.p);
        gt_mse = (X - X_gt).squaredNorm() / X.cols();
        T.translation().noalias() += -T.rotation()*source_mean + target_mean;
        X.colwise() += target_mean;
        norm_x = T.rotation()*norm_x;

        ///save convergence result
        par.convergence_energy = energy;
        par.convergence_gt_mse = gt_mse;
        par.res_trans = T.matrix();

        ///output
        if (par.print_output)
        {
            std::ofstream out_res(par.out_path);
            if (!out_res.is_open())
            {
                std::cout << "Can't open out file " << par.out_path << std::endl;
            }

            ///output time and energy
            out_res.precision(16);
            for (int i = 0; i<total_iter; i++)
            {
                out_res << times[i] << " "<< energys[i] << " " << gt_mses[i] << std::endl;
            }
            out_res.close();
            std::cout << " write res to " << par.out_path << std::endl;
        }
    }



    /// Reweighted ICP with point to plane
    /// @param Source (one 3D point per column)
    /// @param Target (one 3D point per column)
    /// @param Target normals (one 3D normal per column)
    /// @param Parameters
    //    template <typename Derived1, typename Derived2, typename Derived3>
    void point_to_plane_GN(Eigen::Matrix3Xd& X,
                           Eigen::Matrix3Xd& Y, Eigen::Matrix3Xd& norm_x, Eigen::Matrix3Xd& norm_y,
                           Eigen::Vector3d& source_mean, Eigen::Vector3d& target_mean,
                           ICP::Parameters &par) {
        /// Build kd-tree
        KDtree kdtree(Y);
        /// Buffers
        Eigen::Matrix3Xd Qp = Eigen::Matrix3Xd::Zero(3, X.cols());
        Eigen::Matrix3Xd Qn = Eigen::Matrix3Xd::Zero(3, X.cols());
        Eigen::VectorXd W = Eigen::VectorXd::Zero(X.cols());
        Eigen::Matrix3Xd ori_X = X;
        AffineNd T;
        if (par.use_init) T.matrix() = par.init_trans;
        else T = AffineNd::Identity();
        AffineMatrixN To1 = T.matrix();
        X = T*X;

        Eigen::Matrix3Xd X_gt = X;
        if(par.has_groundtruth)
        {
            Eigen::Vector3d temp_trans = par.gt_trans.block(0, 3, 3, 1);
            X_gt = ori_X;
            X_gt.colwise() += source_mean;
            X_gt = par.gt_trans.block(0, 0, 3, 3) * X_gt;
            X_gt.colwise() += temp_trans - target_mean;
        }

        std::vector<double> times, energys, gt_mses;
        double begin_time, end_time, run_time;
        double gt_mse;

        ///dynamic welsch, calc k-nearest points with itself;
        double nu1 = 1, nu2 = 1;
        double begin_init = omp_get_wtime();

        //Anderson Acc para
        AndersonAcceleration accelerator_;
        Vector6 LG_T;
        Vector6 Dir;
        //add time test
        double energy = 0.0, prev_energy = std::numeric_limits<double>::max();
        if(par.use_AA)
        {
            Eigen::Matrix4d log_T = LogMatrix(T.matrix());
            LG_T = LogToVec(log_T);
            accelerator_.init(par.anderson_m, 6, LG_T.data());
        }

        // Find closest point
#pragma omp parallel for
        for (int i = 0; i<X.cols(); ++i) {
            int id = kdtree.closest(X.col(i).data());
            Qp.col(i) = Y.col(id);
            Qn.col(i) = norm_y.col(id);
            W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
        }

        if(par.f == ICP::WELSCH)
        {
            double med1;
            igl::median(W, med1);
            nu1 =par.nu_begin_k * med1;
            nu2 = par.nu_end_k * FindKnearestNormMed(kdtree, Y, 7, norm_y);
            nu1 = nu1>nu2? nu1:nu2;
        }
        double end_init = omp_get_wtime();
        double init_time = end_init - begin_init;

        begin_time = omp_get_wtime();
        int total_iter = 0;
        double test_total_time = 0.0;
        bool stop1 = false;
        par.max_icp = 6;
        while(!stop1)
        {
            par.max_icp = std::min(par.max_icp+1, 10);
            /// ICP
            for(int icp=0; icp<par.max_icp; ++icp) {
                total_iter++;

                int n_linsearch = 0;
                energy = get_energy(par.f, W, nu1);
                if(par.use_AA)
                {
                    if(energy < prev_energy)
                    {
                        prev_energy = energy;
                    }
                    else
                    {
                        // line search
                        double alpha = 0.0;
                        Vector6 new_t = LG_T;
                        Eigen::VectorXd lowest_W = W;
                        Eigen::Matrix3Xd lowest_Qp = Qp;
                        Eigen::Matrix3Xd lowest_Qn = Qn;
                        Eigen::Affine3d lowest_T = T;
                        n_linsearch++;
                        alpha = 1;
                        new_t = LG_T + alpha * Dir;
                        T.matrix() = VecToLog(new_t).exp();
                        /// Find closest point
#pragma omp parallel for
                        for(int i=0; i<X.cols(); i++) {
                            X.col(i) = T * ori_X.col(i);
                            int id = kdtree.closest(X.col(i).data());
                            Qp.col(i) = Y.col(id);
                            Qn.col(i) = norm_y.col(id);
                            W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
                        }
                        double test_energy = get_energy(par.f, W, nu1);
                       if(test_energy < energy)
                        {
                            accelerator_.reset(new_t.data());
                            energy = test_energy;
                        }
                        else
                        {
                            Qp = lowest_Qp;
                            Qn = lowest_Qn;
                            W = lowest_W;
                            T = lowest_T;
                        }
                        prev_energy = energy;
                    }
                }
                else
                {
                    prev_energy = energy;
                }

                end_time = omp_get_wtime();
                run_time = end_time - begin_time;
                energys.push_back(prev_energy);
                times.push_back(run_time);
                if(par.has_groundtruth)
                {
                    gt_mse = (X - X_gt).squaredNorm()/X.cols();
                }
                gt_mses.push_back(gt_mse);

                /// Compute weights
                robust_weight(par.f, W, nu1);
                /// Rotation and translation update
                point_to_plane_gaussnewton(ori_X, Qp, Qn, W, T.matrix(), Dir);
                LG_T = LogToVec(LogMatrix(T.matrix()));
                LG_T += Dir;
                T.matrix() = VecToLog(LG_T).exp();

                // Anderson acc
                if(par.use_AA)
                {
                    Vector6 AA_t;
                    AA_t = accelerator_.compute(LG_T.data());
                    T.matrix() = VecToLog(AA_t).exp();
                }
                if(par.print_energy)
                    std::cout << "icp iter = " << total_iter << ", gt_mse = " << gt_mse
                              << ", nu1 = " << nu1 << ", acept_aa= " << n_linsearch
                              << ", energy = " << prev_energy << std::endl;

                /// Find closest point
#pragma omp parallel for
                for(int i=0; i<X.cols(); i++) {
                    X.col(i) = T * ori_X.col(i);
                    int id = kdtree.closest(X.col(i).data());
                    Qp.col(i) = Y.col(id);
                    Qn.col(i) = norm_y.col(id);
                    W[i] = std::abs(Qn.col(i).transpose() * (X.col(i) - Qp.col(i)));
                }

                /// Stopping criteria
                double stop2 = (T.matrix() - To1).norm();
                To1 = T.matrix();
                if(stop2 < par.stop) break;
            }

            if(par.f == ICP::WELSCH)
            {
                stop1 = fabs(nu1 - nu2)<SAME_THRESHOLD? true: false;
                nu1 = nu1*par.nu_alpha > nu2 ? nu1*par.nu_alpha : nu2;
                if(par.use_AA)
                {
                    accelerator_.reset(LogToVec(LogMatrix(T.matrix())).data());
                    prev_energy = std::numeric_limits<double>::max();
                }
            }
            else
                stop1 = true;
        }

        par.res_trans = T.matrix();

        ///calc convergence energy
        W = (Qn.array()*(X - Qp).array()).colwise().sum().abs().transpose();
        energy = get_energy(par.f, W, nu1);
        gt_mse = (X - X_gt).squaredNorm() / X.cols();
        T.translation().noalias() += -T.rotation()*source_mean + target_mean;
        X.colwise() += target_mean;
        norm_x = T.rotation()*norm_x;

        ///save convergence result
        par.convergence_energy = energy;
        par.convergence_gt_mse = gt_mse;
        par.res_trans = T.matrix();

        ///output
        if (par.print_output)
        {
            std::ofstream out_res(par.out_path);
            if (!out_res.is_open())
            {
                std::cout << "Can't open out file " << par.out_path << std::endl;
            }

            ///output time and energy
            out_res.precision(16);
            for (int i = 0; i<total_iter; i++)
            {
                out_res << times[i] << " "<< energys[i] << " " << gt_mses[i] << std::endl;
            }
            out_res.close();
            std::cout << " write res to " << par.out_path << std::endl;
        }
    }
};


#endif
