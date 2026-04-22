#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace jacobi::svd
{
    /**
     * @brief 布局转置策略；Layout-transpose policy.
     */
    enum class LayoutTransposeMode
    {
        /**
         * @brief 自动按阈值启用；Enable automatically by thresholds.
         */
        auto_select,

        /**
         * @brief 强制启用布局转置；Force-enable layout transpose.
         */
        force_enable,

        /**
         * @brief 强制禁用布局转置；Force-disable layout transpose.
         */
        force_disable
    };

    /**
     * @brief 布局转置阈值自动调优报告；Auto-tuning report for layout-transpose thresholds.
     */
    struct LayoutTransposeAutoTuneReport final
    {
        /**
         * @brief 是否执行过基准扫描；Whether benchmark scan has been executed.
         */
        bool executed = false;

        /**
         * @brief 推荐最小列数阈值；Recommended minimum-column threshold.
         */
        int recommended_min_columns = 16;

        /**
         * @brief 推荐最小元素数阈值；Recommended minimum-element threshold.
         */
        std::size_t recommended_min_elements = 4096;

        /**
         * @brief 估计最优点加速比（direct/transpose）；Estimated best-point speedup ratio (direct/transpose).
         */
        double estimated_best_speedup = 1.0;

        /**
         * @brief 样本扫描数量；Number of scanned samples.
         */
        std::size_t sample_count = 0;
    };

    /**
     * @brief CUDA 运行时错误封装；CUDA runtime error wrapper.
     */
    class CudaError final : public std::runtime_error
    {
    public:
        /**
         * @brief 构造 CUDA 异常对象；Construct a CUDA exception object.
         * @param message 错误消息；Error message.
         */
        explicit CudaError(const char *message);
    };

    /**
     * @brief 单边雅可比 SVD 参数；Configuration for one-sided Jacobi SVD.
     */
    struct JacobiSvdConfig final
    {
        /**
         * @brief 相对收敛阈值 epsilon；Relative convergence tolerance epsilon.
         */
        double epsilon = 1.0e-9;

        /**
         * @brief 最大 sweep 次数；Maximum number of sweeps.
         */
        int max_sweeps = 128;

        /**
         * @brief 每个 CUDA block 的线程数；Threads per CUDA block.
         */
        int threads_per_block = 256;

        /**
         * @brief 布局转置策略；Layout-transpose policy.
         */
        LayoutTransposeMode layout_transpose_mode = LayoutTransposeMode::auto_select;

        /**
         * @brief 自动策略下最小列数阈值；Minimum column threshold when mode is auto.
         */
        int layout_transpose_min_columns = 16;

        /**
         * @brief 自动策略下最小元素数阈值；Minimum element threshold when mode is auto.
         */
        std::size_t layout_transpose_min_elements = 4096;

        /**
         * @brief 是否在运行前执行阈值微基准自动调优；Whether to run micro-benchmark auto-tuning before execution.
         */
        bool layout_transpose_auto_tune = false;

        /**
         * @brief 自动调优时每个尺寸的重复次数；Repetition count per size during auto-tuning.
         */
        int layout_transpose_benchmark_repetitions = 2;

        /**
         * @brief 自动调优时的基准 sweep 上限；Benchmark sweep cap during auto-tuning.
         */
        int layout_transpose_benchmark_sweeps = 8;
    };

    /**
     * @brief 行主序设备矩阵封装；Row-major device matrix wrapper.
     * @note 该类型负责 GPU 内存生命周期，kernel 侧仅接收裸指针；This type owns GPU memory lifecycle and kernels only receive raw pointers.
     */
    class DeviceMatrix final
    {
    public:
        /**
         * @brief 默认构造空矩阵；Default construct an empty matrix.
         */
        DeviceMatrix() = default;

        /**
         * @brief 构造并分配设备矩阵；Construct and allocate a device matrix.
         * @param rows 行数；Row count.
         * @param columns 列数；Column count.
         */
        DeviceMatrix(std::size_t rows, std::size_t columns);

        /**
         * @brief 析构并释放设备内存；Destroy and release device memory.
         */
        ~DeviceMatrix();

        /**
         * @brief 禁止拷贝构造；Copy construction is disabled.
         */
        DeviceMatrix(const DeviceMatrix &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        DeviceMatrix &operator=(const DeviceMatrix &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        DeviceMatrix(DeviceMatrix &&other) noexcept;

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        DeviceMatrix &operator=(DeviceMatrix &&other) noexcept;

        /**
         * @brief 重新分配矩阵尺寸；Reallocate matrix with new shape.
         * @param rows 行数；Row count.
         * @param columns 列数；Column count.
         */
        void reset(std::size_t rows, std::size_t columns);

        /**
         * @brief 读取行数；Get number of rows.
         * @return 行数；Row count.
         */
        [[nodiscard]] std::size_t rows() const noexcept;

        /**
         * @brief 读取列数；Get number of columns.
         * @return 列数；Column count.
         */
        [[nodiscard]] std::size_t columns() const noexcept;

        /**
         * @brief 元素总数；Total number of elements.
         * @return 元素数；Element count.
         */
        [[nodiscard]] std::size_t size() const noexcept;

        /**
         * @brief 字节总数；Total bytes.
         * @return 字节数；Byte count.
         */
        [[nodiscard]] std::size_t bytes() const noexcept;

        /**
         * @brief 获取可写裸指针；Get mutable raw pointer.
         * @return 设备指针；Device pointer.
         */
        [[nodiscard]] double *data() noexcept;

        /**
         * @brief 获取只读裸指针；Get const raw pointer.
         * @return 设备指针；Device pointer.
         */
        [[nodiscard]] const double *data() const noexcept;

        /**
         * @brief 将主机数据拷贝到设备；Copy host data into device matrix.
         * @param host_values 主机行主序数据；Host row-major values.
         */
        void copy_from_host(std::span<const double> host_values);

        /**
         * @brief 将设备数据拷贝回主机；Copy device data back to host.
         * @return 主机行主序数据；Host row-major values.
         */
        [[nodiscard]] std::vector<double> copy_to_host() const;

    private:
        /**
         * @brief 行数成员；Row count field.
         */
        std::size_t rows_ = 0;

        /**
         * @brief 列数成员；Column count field.
         */
        std::size_t columns_ = 0;

        /**
         * @brief 设备数据指针；Device data pointer.
         */
        double *data_ = nullptr;
    };

    /**
     * @brief Jacobi SVD 主机结果容器；Host-side result container for Jacobi SVD.
     */
    struct JacobiSvdResult final
    {
        /**
         * @brief 输入矩阵行数 m；Input row count m.
         */
        std::size_t rows = 0;

        /**
         * @brief 输入矩阵列数 n；Input column count n.
         */
        std::size_t columns = 0;

        /**
         * @brief 左奇异矩阵 U（m x n，行主序）；Left singular matrix U (m x n, row-major).
         */
        std::vector<double> u;

        /**
         * @brief 奇异值向量 Sigma（长度 n）；Singular values Sigma (length n).
         */
        std::vector<double> sigma;

        /**
         * @brief 右奇异矩阵 V（n x n，行主序）；Right singular matrix V (n x n, row-major).
         */
        std::vector<double> v;

        /**
         * @brief 实际执行 sweep 次数；Number of executed sweeps.
         */
        int sweeps = 0;
    };

    /**
     * @brief 执行单边雅可比奇异值分解；Run one-sided Jacobi singular value decomposition.
     * @param host_input 输入矩阵 A（行主序，m x n）；Input matrix A (row-major, m x n).
     * @param rows 输入行数 m；Input row count m.
     * @param columns 输入列数 n；Input column count n.
     * @param config 算法配置；Algorithm configuration.
     * @return 主机侧 SVD 结果；Host-side SVD result.
     * @note 当前实现假设 m >= n；Current implementation assumes m >= n.
     * @example
     * // 中文：输入 3x2 矩阵，输出 U、Sigma、V。
     * // English: Decompose a 3x2 matrix into U, Sigma, and V.
     * // auto result = jacobi::svd::one_sided_jacobi_svd(a, 3, 2, {});
     */
    [[nodiscard]] JacobiSvdResult one_sided_jacobi_svd(std::span<const double> host_input,
                                                       std::size_t rows,
                                                       std::size_t columns,
                                                       const JacobiSvdConfig &config = {});

    /**
     * @brief 自动扫描矩阵尺寸并给出布局转置阈值建议；Scan matrix sizes and recommend layout-transpose thresholds.
     * @param config 基准配置模板；Template configuration for benchmark.
     * @return 阈值自动调优报告；Threshold auto-tuning report.
     * @note 该函数在领域层执行微基准，不修改全局状态；This function runs micro-benchmark in domain layer without mutating global state.
     */
    [[nodiscard]] LayoutTransposeAutoTuneReport auto_tune_layout_transpose_threshold(
        const JacobiSvdConfig &config = {});
} // namespace jacobi::svd
