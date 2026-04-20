#include "kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace jacobi::svd
{
    /**
     * @brief 将 CUDA 错误码转换为异常；Convert CUDA status to exception.
     * @param status CUDA 返回码；CUDA return status.
     * @param expression 触发检查的表达式文本；Checked expression text.
     * @param file 源文件名；Source file name.
     * @param line 源码行号；Source line number.
     */
    inline void throw_if_cuda_failed(cudaError_t status, const char *expression, const char *file, int line)
    {
        if (status == cudaSuccess)
        {
            return;
        }

        std::ostringstream stream;
        stream << "CUDA call failed: " << expression << " @ " << file << ':' << line
               << ", code=" << static_cast<int>(status) << ", message=" << cudaGetErrorString(status);
        throw CudaError(stream.str().c_str());
    }
} // namespace jacobi::svd

/**
 * @brief CUDA 调用检查宏；CUDA call checking macro.
 */
#define JACOBI_CUDA_CHECK(EXPR) ::jacobi::svd::throw_if_cuda_failed((EXPR), #EXPR, __FILE__, __LINE__)

namespace jacobi::svd
{
    namespace
    {
        /**
         * @brief 设备缓冲区 RAII 封装；RAII wrapper for device buffers.
         * @tparam T 元素类型；Element type.
         */
        template <typename T>
        class DeviceBuffer final
        {
        public:
            /**
             * @brief 默认构造空缓冲区；Default construct an empty buffer.
             */
            DeviceBuffer() = default;

            /**
             * @brief 构造并分配缓冲区；Construct and allocate buffer.
             * @param count 元素数量；Element count.
             */
            explicit DeviceBuffer(std::size_t count)
            {
                reset(count);
            }

            /**
             * @brief 析构释放资源；Destroy and release resource.
             */
            ~DeviceBuffer()
            {
                release();
            }

            /**
             * @brief 禁止拷贝；Copy is disabled.
             */
            DeviceBuffer(const DeviceBuffer &) = delete;

            /**
             * @brief 禁止拷贝赋值；Copy assignment is disabled.
             * @return 当前对象引用；Reference to current object.
             */
            DeviceBuffer &operator=(const DeviceBuffer &) = delete;

            /**
             * @brief 移动构造；Move constructor.
             * @param other 源缓冲区；Source buffer.
             */
            DeviceBuffer(DeviceBuffer &&other) noexcept
                : count_(std::exchange(other.count_, 0)), data_(std::exchange(other.data_, nullptr))
            {
            }

            /**
             * @brief 移动赋值；Move assignment.
             * @param other 源缓冲区；Source buffer.
             * @return 当前对象引用；Reference to current object.
             */
            DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
            {
                if (this != &other)
                {
                    release();
                    count_ = std::exchange(other.count_, 0);
                    data_ = std::exchange(other.data_, nullptr);
                }
                return *this;
            }

            /**
             * @brief 重新分配缓冲区；Reallocate buffer.
             * @param count 元素数量；Element count.
             */
            void reset(std::size_t count)
            {
                release();
                count_ = count;
                if (count_ == 0)
                {
                    return;
                }
                JACOBI_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), count_ * sizeof(T)));
            }

            /**
             * @brief 获取数据指针；Get data pointer.
             * @return 设备指针；Device pointer.
             */
            [[nodiscard]] T *data() noexcept
            {
                return data_;
            }

            /**
             * @brief 获取常量数据指针；Get const data pointer.
             * @return 设备指针；Device pointer.
             */
            [[nodiscard]] const T *data() const noexcept
            {
                return data_;
            }

            /**
             * @brief 获取元素数量；Get element count.
             * @return 元素数量；Element count.
             */
            [[nodiscard]] std::size_t size() const noexcept
            {
                return count_;
            }

        private:
            /**
             * @brief 释放设备内存；Release device memory.
             */
            void release() noexcept
            {
                if (data_ != nullptr)
                {
                    (void)cudaFree(data_);
                    data_ = nullptr;
                }
                count_ = 0;
            }

            /**
             * @brief 元素数量；Element count.
             */
            std::size_t count_ = 0;

            /**
             * @brief 设备数据指针；Device data pointer.
             */
            T *data_ = nullptr;
        };

        /**
         * @brief 行主序索引映射；Row-major index mapping.
         * @param row 行索引；Row index.
         * @param col 列索引；Column index.
         * @param columns 总列数；Total columns.
         * @return 一维偏移；Linear offset.
         */
        __host__ __device__ inline int row_major_index(int row, int col, int columns)
        {
            return row * columns + col;
        }

        /**
         * @brief 初始化单位矩阵 V；Initialize identity matrix V.
         * @param v 输出矩阵指针；Output matrix pointer.
         * @param n 方阵维度；Square dimension.
         */
        __global__ void initialize_identity_kernel(double *v, int n)
        {
            const int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int total = n * n;
            if (linear >= total)
            {
                return;
            }

            const int row = linear / n;
            const int col = linear % n;
            v[linear] = (row == col) ? 1.0 : 0.0;
        }

        /**
         * @brief 计算列对统计量；Compute pair statistics for column pairs.
         * @param a 输入矩阵 A；Input matrix A.
         * @param m 行数；Row count.
         * @param n 列数；Column count.
         * @param pairs 列对数组；Column-pair array.
         * @param pair_count 列对数量；Number of pairs.
         * @param a_pp 输出 A_p 点积；Output A_p dot A_p.
         * @param a_qq 输出 A_q 点积；Output A_q dot A_q.
         * @param a_pq 输出 A_p 与 A_q 点积；Output A_p dot A_q.
         */
        __global__ void pair_stats_kernel(const double *a,
                                          int m,
                                          int n,
                                          const int2 *pairs,
                                          int pair_count,
                                          double *a_pp,
                                          double *a_qq,
                                          double *a_pq)
        {
            const int pair_index = static_cast<int>(blockIdx.x);
            if (pair_index >= pair_count)
            {
                return;
            }

            const int tid = static_cast<int>(threadIdx.x);
            const int p = pairs[pair_index].x;
            const int q = pairs[pair_index].y;

            double local_pp = 0.0;
            double local_qq = 0.0;
            double local_pq = 0.0;

            for (int row = tid; row < m; row += static_cast<int>(blockDim.x))
            {
                const double ap = a[row_major_index(row, p, n)];
                const double aq = a[row_major_index(row, q, n)];
                local_pp += ap * ap;
                local_qq += aq * aq;
                local_pq += ap * aq;
            }

            extern __shared__ double shared[];
            double *shared_pp = shared;
            double *shared_qq = shared + blockDim.x;
            double *shared_pq = shared + (2 * blockDim.x);

            shared_pp[tid] = local_pp;
            shared_qq[tid] = local_qq;
            shared_pq[tid] = local_pq;
            __syncthreads();

            for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1U)
            {
                if (tid < static_cast<int>(stride))
                {
                    shared_pp[tid] += shared_pp[tid + stride];
                    shared_qq[tid] += shared_qq[tid + stride];
                    shared_pq[tid] += shared_pq[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0)
            {
                a_pp[pair_index] = shared_pp[0];
                a_qq[pair_index] = shared_qq[0];
                a_pq[pair_index] = shared_pq[0];
            }
        }

        /**
         * @brief 计算旋转参数并判断是否需要旋转；Compute rotation parameters and decide whether to rotate.
         * @param a_pp A_p 点积数组；A_p dot A_p array.
         * @param a_qq A_q 点积数组；A_q dot A_q array.
         * @param a_pq A_p 与 A_q 点积数组；A_p dot A_q array.
         * @param pair_count 列对数量；Number of pairs.
         * @param epsilon 收敛阈值；Convergence threshold.
         * @param c 输出 cos(theta) 数组；Output cos(theta) array.
         * @param s 输出 sin(theta) 数组；Output sin(theta) array.
         * @param any_rotation_flag 若发生旋转则置 1；Set to 1 if any rotation happens.
         */
        __global__ void compute_rotation_params_kernel(const double *a_pp,
                                                       const double *a_qq,
                                                       const double *a_pq,
                                                       int pair_count,
                                                       double epsilon,
                                                       double *c,
                                                       double *s,
                                                       int *any_rotation_flag)
        {
            const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (index >= pair_count)
            {
                return;
            }

            const double pp = a_pp[index];
            const double qq = a_qq[index];
            const double pq = a_pq[index];
            const double rhs = epsilon * sqrt(fmax(pp, 0.0) * fmax(qq, 0.0));

            if (fabs(pq) > rhs && fabs(pq) > 1.0e-300)
            {
                const double tau = (qq - pp) / (2.0 * pq);
                const double t = (tau >= 0.0) ? (1.0 / (tau + sqrt(1.0 + tau * tau)))
                                              : (-1.0 / (-tau + sqrt(1.0 + tau * tau)));
                const double cosine = 1.0 / sqrt(1.0 + t * t);
                const double sine = t * cosine;
                c[index] = cosine;
                s[index] = sine;
                atomicExch(any_rotation_flag, 1);
            }
            else
            {
                c[index] = 1.0;
                s[index] = 0.0;
            }
        }

        /**
         * @brief 应用 Givens 旋转到 A 与 V；Apply Givens rotation to A and V.
         * @param a 输入输出矩阵 A；Input/output matrix A.
         * @param v 输入输出矩阵 V；Input/output matrix V.
         * @param m A 的行数；Row count of A.
         * @param n A 的列数，同时也是 V 的维度；Column count of A and dimension of V.
         * @param pairs 列对数组；Column-pair array.
         * @param pair_count 列对数量；Number of pairs.
         * @param c cos(theta) 数组；cos(theta) array.
         * @param s sin(theta) 数组；sin(theta) array.
         */
        __global__ void apply_rotation_kernel(double *a,
                                              double *v,
                                              int m,
                                              int n,
                                              const int2 *pairs,
                                              int pair_count,
                                              const double *c,
                                              const double *s)
        {
            const int pair_index = static_cast<int>(blockIdx.x);
            if (pair_index >= pair_count)
            {
                return;
            }

            const int tid = static_cast<int>(threadIdx.x);
            const int p = pairs[pair_index].x;
            const int q = pairs[pair_index].y;
            const double cosine = c[pair_index];
            const double sine = s[pair_index];

            for (int row = tid; row < m; row += static_cast<int>(blockDim.x))
            {
                const int idx_p = row_major_index(row, p, n);
                const int idx_q = row_major_index(row, q, n);
                const double value_p = a[idx_p];
                const double value_q = a[idx_q];
                a[idx_p] = cosine * value_p - sine * value_q;
                a[idx_q] = sine * value_p + cosine * value_q;
            }

            for (int row = tid; row < n; row += static_cast<int>(blockDim.x))
            {
                const int idx_p = row_major_index(row, p, n);
                const int idx_q = row_major_index(row, q, n);
                const double value_p = v[idx_p];
                const double value_q = v[idx_q];
                v[idx_p] = cosine * value_p - sine * value_q;
                v[idx_q] = sine * value_p + cosine * value_q;
            }
        }

        /**
         * @brief 从收敛后的 A 构建 U 与 Sigma；Build U and Sigma from converged A.
         * @param a 输入矩阵 A（应为 U*Sigma）；Input matrix A (should be U*Sigma).
         * @param u 输出矩阵 U；Output matrix U.
         * @param sigma 输出奇异值数组；Output singular values.
         * @param m 行数；Row count.
         * @param n 列数；Column count.
         * @param epsilon 避免除零的阈值；Threshold to avoid divide-by-zero.
         */
        __global__ void build_u_sigma_kernel(const double *a, double *u, double *sigma, int m, int n, double epsilon)
        {
            const int col = static_cast<int>(blockIdx.x);
            if (col >= n)
            {
                return;
            }

            const int tid = static_cast<int>(threadIdx.x);
            double local_norm = 0.0;

            for (int row = tid; row < m; row += static_cast<int>(blockDim.x))
            {
                const double value = a[row_major_index(row, col, n)];
                local_norm += value * value;
            }

            extern __shared__ double shared_norm[];
            shared_norm[tid] = local_norm;
            __syncthreads();

            for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1U)
            {
                if (tid < static_cast<int>(stride))
                {
                    shared_norm[tid] += shared_norm[tid + stride];
                }
                __syncthreads();
            }

            __shared__ double sigma_col;
            if (tid == 0)
            {
                sigma_col = sqrt(fmax(shared_norm[0], 0.0));
                sigma[col] = sigma_col;
            }
            __syncthreads();

            for (int row = tid; row < m; row += static_cast<int>(blockDim.x))
            {
                const int index = row_major_index(row, col, n);
                u[index] = (sigma_col > epsilon) ? (a[index] / sigma_col) : 0.0;
            }
        }

        /**
         * @brief 构建巡回赛无冲突列对调度；Build conflict-free round-robin column-pair schedule.
         * @param columns 列数 n；Number of columns n.
         * @return 每个 round 的列对集合；Column-pair groups per round.
         * @note 使用固定首元素的轮转法，保证同一 round 内列不重复；Uses fixed-head rotation to ensure disjoint pairs in each round.
         */
        [[nodiscard]] std::vector<std::vector<int2>> build_round_robin_schedule(int columns)
        {
            if (columns < 2)
            {
                return {};
            }

            const bool needs_dummy = (columns % 2) != 0;
            const int even_columns = needs_dummy ? (columns + 1) : columns;

            std::vector<int> circle(static_cast<std::size_t>(even_columns), -1);
            for (int index = 0; index < columns; ++index)
            {
                circle[static_cast<std::size_t>(index)] = index;
            }

            std::vector<std::vector<int2>> rounds(static_cast<std::size_t>(even_columns - 1));
            for (int round = 0; round < even_columns - 1; ++round)
            {
                auto &pairs = rounds[static_cast<std::size_t>(round)];
                pairs.reserve(static_cast<std::size_t>(even_columns / 2));

                for (int i = 0; i < even_columns / 2; ++i)
                {
                    const int lhs = circle[static_cast<std::size_t>(i)];
                    const int rhs = circle[static_cast<std::size_t>(even_columns - 1 - i)];
                    if (lhs >= 0 && rhs >= 0)
                    {
                        pairs.push_back(make_int2(lhs, rhs));
                    }
                }

                const int last = circle[static_cast<std::size_t>(even_columns - 1)];
                for (int i = even_columns - 1; i > 1; --i)
                {
                    circle[static_cast<std::size_t>(i)] = circle[static_cast<std::size_t>(i - 1)];
                }
                circle[1] = last;
            }

            return rounds;
        }

        /**
         * @brief 规范化线程数量到合法 CUDA 配置；Normalize thread count to valid CUDA launch size.
         * @param raw_threads 用户配置线程数；User-configured thread count.
         * @return 合法且按 warp 对齐的线程数；Valid warp-aligned thread count.
         */
        [[nodiscard]] int normalize_threads_per_block(int raw_threads)
        {
            const int clamped = std::clamp(raw_threads, 32, 1024);
            const int aligned = ((clamped + 31) / 32) * 32;
            return std::min(aligned, 1024);
        }
    } // namespace

    CudaError::CudaError(const char *message)
        : std::runtime_error(message)
    {
    }

    DeviceMatrix::DeviceMatrix(std::size_t rows, std::size_t columns)
    {
        reset(rows, columns);
    }

    DeviceMatrix::~DeviceMatrix()
    {
        if (data_ != nullptr)
        {
            (void)cudaFree(data_);
            data_ = nullptr;
        }
    }

    DeviceMatrix::DeviceMatrix(DeviceMatrix &&other) noexcept
        : rows_(std::exchange(other.rows_, 0)),
          columns_(std::exchange(other.columns_, 0)),
          data_(std::exchange(other.data_, nullptr))
    {
    }

    DeviceMatrix &DeviceMatrix::operator=(DeviceMatrix &&other) noexcept
    {
        if (this != &other)
        {
            if (data_ != nullptr)
            {
                (void)cudaFree(data_);
            }
            rows_ = std::exchange(other.rows_, 0);
            columns_ = std::exchange(other.columns_, 0);
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }

    void DeviceMatrix::reset(std::size_t rows, std::size_t columns)
    {
        if (rows == 0 || columns == 0)
        {
            throw std::invalid_argument("DeviceMatrix dimensions must be positive.");
        }

        if (data_ != nullptr)
        {
            JACOBI_CUDA_CHECK(cudaFree(data_));
            data_ = nullptr;
        }

        rows_ = rows;
        columns_ = columns;
        JACOBI_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), bytes()));
    }

    std::size_t DeviceMatrix::rows() const noexcept
    {
        return rows_;
    }

    std::size_t DeviceMatrix::columns() const noexcept
    {
        return columns_;
    }

    std::size_t DeviceMatrix::size() const noexcept
    {
        return rows_ * columns_;
    }

    std::size_t DeviceMatrix::bytes() const noexcept
    {
        return size() * sizeof(double);
    }

    double *DeviceMatrix::data() noexcept
    {
        return data_;
    }

    const double *DeviceMatrix::data() const noexcept
    {
        return data_;
    }

    void DeviceMatrix::copy_from_host(std::span<const double> host_values)
    {
        if (host_values.size() != size())
        {
            throw std::invalid_argument("Host data size does not match device matrix shape.");
        }
        JACOBI_CUDA_CHECK(cudaMemcpy(data_, host_values.data(), bytes(), cudaMemcpyHostToDevice));
    }

    std::vector<double> DeviceMatrix::copy_to_host() const
    {
        std::vector<double> host(size());
        JACOBI_CUDA_CHECK(cudaMemcpy(host.data(), data_, bytes(), cudaMemcpyDeviceToHost));
        return host;
    }

    JacobiSvdResult one_sided_jacobi_svd(std::span<const double> host_input,
                                         std::size_t rows,
                                         std::size_t columns,
                                         const JacobiSvdConfig &config)
    {
        if (rows == 0 || columns == 0)
        {
            throw std::invalid_argument("Input matrix shape must be non-zero.");
        }
        if (rows < columns)
        {
            throw std::invalid_argument("One-sided Jacobi SVD requires rows >= columns in this implementation.");
        }
        if (host_input.size() != rows * columns)
        {
            throw std::invalid_argument("Input buffer size mismatch.");
        }
        if (config.epsilon <= 0.0)
        {
            throw std::invalid_argument("epsilon must be positive.");
        }
        if (config.max_sweeps <= 0)
        {
            throw std::invalid_argument("max_sweeps must be positive.");
        }

        const int m = static_cast<int>(rows);
        const int n = static_cast<int>(columns);
        const int threads = normalize_threads_per_block(config.threads_per_block);

        DeviceMatrix d_a(rows, columns);
        DeviceMatrix d_v(columns, columns);
        DeviceMatrix d_u(rows, columns);
        DeviceMatrix d_sigma(1, columns);

        d_a.copy_from_host(host_input);

        const int identity_total = n * n;
        const int identity_blocks = (identity_total + threads - 1) / threads;
        initialize_identity_kernel<<<identity_blocks, threads>>>(d_v.data(), n);
        JACOBI_CUDA_CHECK(cudaGetLastError());

        const auto rounds = build_round_robin_schedule(n);
        std::size_t max_pairs = 0;
        for (const auto &round : rounds)
        {
            max_pairs = std::max(max_pairs, round.size());
        }

        DeviceBuffer<int2> d_pairs(max_pairs);
        DeviceBuffer<double> d_app(max_pairs);
        DeviceBuffer<double> d_aqq(max_pairs);
        DeviceBuffer<double> d_apq(max_pairs);
        DeviceBuffer<double> d_c(max_pairs);
        DeviceBuffer<double> d_s(max_pairs);
        DeviceBuffer<int> d_any_rotation(1);

        int executed_sweeps = 0;

        for (int sweep = 0; sweep < config.max_sweeps; ++sweep)
        {
            bool converged_this_sweep = true;

            for (const auto &round : rounds)
            {
                const int pair_count = static_cast<int>(round.size());
                if (pair_count == 0)
                {
                    continue;
                }

                JACOBI_CUDA_CHECK(cudaMemcpy(d_pairs.data(),
                                             round.data(),
                                             static_cast<std::size_t>(pair_count) * sizeof(int2),
                                             cudaMemcpyHostToDevice));
                JACOBI_CUDA_CHECK(cudaMemset(d_any_rotation.data(), 0, sizeof(int)));

                const std::size_t shared_bytes_stats = static_cast<std::size_t>(threads) * 3 * sizeof(double);
                pair_stats_kernel<<<pair_count, threads, shared_bytes_stats>>>(d_a.data(),
                                                                               m,
                                                                               n,
                                                                               d_pairs.data(),
                                                                               pair_count,
                                                                               d_app.data(),
                                                                               d_aqq.data(),
                                                                               d_apq.data());
                JACOBI_CUDA_CHECK(cudaGetLastError());

                const int rotation_blocks = (pair_count + threads - 1) / threads;
                compute_rotation_params_kernel<<<rotation_blocks, threads>>>(d_app.data(),
                                                                             d_aqq.data(),
                                                                             d_apq.data(),
                                                                             pair_count,
                                                                             config.epsilon,
                                                                             d_c.data(),
                                                                             d_s.data(),
                                                                             d_any_rotation.data());
                JACOBI_CUDA_CHECK(cudaGetLastError());

                apply_rotation_kernel<<<pair_count, threads>>>(d_a.data(),
                                                               d_v.data(),
                                                               m,
                                                               n,
                                                               d_pairs.data(),
                                                               pair_count,
                                                               d_c.data(),
                                                               d_s.data());
                JACOBI_CUDA_CHECK(cudaGetLastError());

                int any_rotation = 0;
                JACOBI_CUDA_CHECK(cudaMemcpy(&any_rotation, d_any_rotation.data(), sizeof(int), cudaMemcpyDeviceToHost));
                if (any_rotation != 0)
                {
                    converged_this_sweep = false;
                }
            }

            executed_sweeps = sweep + 1;
            if (converged_this_sweep)
            {
                break;
            }
        }

        const std::size_t shared_bytes_norm = static_cast<std::size_t>(threads) * sizeof(double);
        build_u_sigma_kernel<<<n, threads, shared_bytes_norm>>>(
            d_a.data(), d_u.data(), d_sigma.data(), m, n, config.epsilon);
        JACOBI_CUDA_CHECK(cudaGetLastError());
        JACOBI_CUDA_CHECK(cudaDeviceSynchronize());

        JacobiSvdResult result;
        result.rows = rows;
        result.columns = columns;
        result.sweeps = executed_sweeps;
        result.u = d_u.copy_to_host();
        result.sigma = d_sigma.copy_to_host();
        result.v = d_v.copy_to_host();
        return result;
    }
} // namespace jacobi::svd

#undef JACOBI_CUDA_CHECK
