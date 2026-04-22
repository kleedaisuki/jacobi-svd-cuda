#pragma once

#include "kernels.hpp"

#include <cstddef>
#include <filesystem>

namespace jacobi::svd::pipeline
{
    /**
     * @brief 矩阵文件格式枚举；Matrix file format enumeration.
     */
    enum class MatrixFileFormat
    {
        /**
         * @brief 自动按扩展名识别（.mat/.txt）；Auto-detect from file extension (.mat/.txt).
         */
        auto_detect,

        /**
         * @brief 二进制 *.mat；Binary *.mat.
         */
        mat,

        /**
         * @brief 文本 *.txt；Text *.txt.
         */
        txt
    };

    /**
     * @brief Pipeline 运行配置；Pipeline runtime configuration.
     */
    struct PipelineConfig final
    {
        /**
         * @brief 测试用例输入路径；Input path of testcase stream.
         */
        std::filesystem::path input_path;

        /**
         * @brief 结果输出路径；Output path of result stream.
         */
        std::filesystem::path output_path;

        /**
         * @brief 输入格式；Input format.
         */
        MatrixFileFormat input_format = MatrixFileFormat::auto_detect;

        /**
         * @brief 输出格式；Output format.
         */
        MatrixFileFormat output_format = MatrixFileFormat::auto_detect;

        /**
         * @brief 输出队列容量（生产者-消费者）；Output queue capacity (producer-consumer).
         */
        std::size_t max_queued_results = 4;

        /**
         * @brief Jacobi SVD 核函数配置；Jacobi SVD kernel configuration.
         */
        JacobiSvdConfig kernel_config{};
    };

    /**
     * @brief Pipeline 执行报告；Pipeline execution report.
     */
    struct PipelineReport final
    {
        /**
         * @brief 已处理测试用例数量；Number of processed testcases.
         */
        std::size_t testcase_count = 0;

        /**
         * @brief 已写出矩阵数量（每例固定 3 张：U/Sigma/V）；Number of emitted matrices (3 per case: U/Sigma/V).
         */
        std::size_t emitted_matrix_count = 0;

        /**
         * @brief 全部测试用例 sweep 总和；Total sweeps across all testcases.
         */
        std::size_t total_sweeps = 0;

        /**
         * @brief 运行时布局转置策略；Runtime layout-transpose policy.
         */
        LayoutTransposeMode layout_transpose_mode = LayoutTransposeMode::auto_select;

        /**
         * @brief 运行时布局转置最小列数阈值；Runtime minimum-column threshold for layout transpose.
         */
        int layout_transpose_min_columns = 16;

        /**
         * @brief 运行时布局转置最小元素阈值；Runtime minimum-element threshold for layout transpose.
         */
        std::size_t layout_transpose_min_elements = 4096;

        /**
         * @brief 本次是否执行了阈值自动调优；Whether threshold auto-tuning was executed in this run.
         */
        bool layout_transpose_auto_tuned = false;

        /**
         * @brief 自动调优估计最优点加速比；Estimated best-point speedup from auto-tuning.
         */
        double layout_transpose_estimated_best_speedup = 1.0;
    };

    /**
     * @brief Jacobi SVD 应用层 Pipeline 聚合根；Application-layer aggregate root for Jacobi SVD pipeline.
     * @note 负责将 testcases -> kernel -> output 三个步骤组装为一个对象；Composes testcases -> kernel -> output into one aggregate object.
     */
    class JacobiSvdPipeline final
    {
    public:
        /**
         * @brief 构造 Pipeline；Construct pipeline.
         * @param config 运行配置；Runtime configuration.
         */
        explicit JacobiSvdPipeline(PipelineConfig config);

        /**
         * @brief 执行 Pipeline；Execute pipeline.
         * @return 执行报告；Execution report.
         * @note 输出流按每个测试用例写出三张矩阵：U、Sigma(1xn)、V；For each testcase, output stream writes U, Sigma(1xn), and V in order.
         */
        [[nodiscard]] PipelineReport run() const;

    private:
        /**
         * @brief 配置对象；Configuration object.
         */
        PipelineConfig config_;
    };

    /**
     * @brief 便捷函数：执行 Jacobi SVD Pipeline；Convenience function to run Jacobi SVD pipeline.
     * @param config 运行配置；Runtime configuration.
     * @return 执行报告；Execution report.
     */
    [[nodiscard]] PipelineReport run_pipeline(const PipelineConfig &config);
} // namespace jacobi::svd::pipeline
