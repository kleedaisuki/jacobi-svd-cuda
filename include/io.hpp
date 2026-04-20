#pragma once

#include <cstddef>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <span>
#include <utility>
#include <vector>

namespace jacobi::svd::io
{
    /**
     * @brief 行主序矩阵容器；Row-major matrix container.
     */
    struct Matrix final
    {
        /**
         * @brief 矩阵行数；Matrix row count.
         */
        std::size_t rows = 0;

        /**
         * @brief 矩阵列数；Matrix column count.
         */
        std::size_t columns = 0;

        /**
         * @brief 行主序元素数据；Row-major element values.
         */
        std::vector<double> values;
    };

    /**
     * @brief *.mat 元数据头；Metadata header for *.mat.
     * @note 文件中使用网络字节序（Network Byte Order）；Network byte order is used on disk.
     */
    struct MatMetaData final
    {
        /**
         * @brief 行数（64 位无符号整数）；Row count (64-bit unsigned integer).
         */
        std::uint64_t rows = 0;

        /**
         * @brief 列数（64 位无符号整数）；Column count (64-bit unsigned integer).
         */
        std::uint64_t columns = 0;
    };

    /**
     * @brief 矩阵读取 policy 概念；Concept for matrix reading policy.
     * @tparam Policy policy 类型；Policy type.
     */
    template <typename Policy>
    concept MatrixReadPolicy = requires(const std::filesystem::path &path)
    {
        {
            Policy::read(path)
        } -> std::same_as<std::vector<Matrix>>;
    };

    /**
     * @brief 矩阵写入 policy 概念；Concept for matrix writing policy.
     * @tparam Policy policy 类型；Policy type.
     */
    template <typename Policy>
    concept MatrixWritePolicy = requires(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        {
            Policy::write(path, matrices)
        } -> std::same_as<void>;
    };

    /**
     * @brief 矩阵输入流模板；Matrix input stream template.
     * @tparam Policy 读取 policy；Read policy.
     */
    template <MatrixReadPolicy Policy>
    class MatrixInputStream final
    {
    public:
        /**
         * @brief 构造输入流；Construct input stream.
         * @param path 文件路径；File path.
         */
        explicit MatrixInputStream(std::filesystem::path path)
            : path_(std::move(path))
        {
        }

        /**
         * @brief 读取全部矩阵；Read all matrices.
         * @return 矩阵序列；Matrix sequence.
         */
        [[nodiscard]] std::vector<Matrix> read_all() const
        {
            return Policy::read(path_);
        }

    private:
        /**
         * @brief 文件路径；File path.
         */
        std::filesystem::path path_;
    };

    /**
     * @brief 矩阵输出流模板；Matrix output stream template.
     * @tparam Policy 写入 policy；Write policy.
     */
    template <MatrixWritePolicy Policy>
    class MatrixOutputStream final
    {
    public:
        /**
         * @brief 构造输出流；Construct output stream.
         * @param path 文件路径；File path.
         */
        explicit MatrixOutputStream(std::filesystem::path path)
            : path_(std::move(path))
        {
        }

        /**
         * @brief 写入全部矩阵；Write all matrices.
         * @param matrices 待写入矩阵序列；Matrix sequence to write.
         */
        void write_all(std::span<const Matrix> matrices) const
        {
            Policy::write(path_, matrices);
        }

    private:
        /**
         * @brief 文件路径；File path.
         */
        std::filesystem::path path_;
    };

    /**
     * @brief *.mat 文件 policy；Policy for *.mat files.
     */
    struct MatFilePolicy final
    {
        /**
         * @brief 从 *.mat 读取矩阵流；Read matrix stream from *.mat.
         * @param path 输入路径；Input path.
         * @return 矩阵序列；Matrix sequence.
         */
        [[nodiscard]] static std::vector<Matrix> read(const std::filesystem::path &path);

        /**
         * @brief 将矩阵流写入 *.mat；Write matrix stream to *.mat.
         * @param path 输出路径；Output path.
         * @param matrices 矩阵序列；Matrix sequence.
         */
        static void write(const std::filesystem::path &path, std::span<const Matrix> matrices);
    };

    /**
     * @brief 文本矩阵文件 policy；Policy for text matrix files.
     * @note 行内以空格分隔，行间以换行分隔，矩阵之间以空行分隔；Values are space-separated, rows are newline-separated, matrices are separated by blank lines.
     */
    struct TxtFilePolicy final
    {
        /**
         * @brief 从文本读取矩阵流；Read matrix stream from text.
         * @param path 输入路径；Input path.
         * @return 矩阵序列；Matrix sequence.
         */
        [[nodiscard]] static std::vector<Matrix> read(const std::filesystem::path &path);

        /**
         * @brief 将矩阵流写入文本；Write matrix stream to text.
         * @param path 输出路径；Output path.
         * @param matrices 矩阵序列；Matrix sequence.
         */
        static void write(const std::filesystem::path &path, std::span<const Matrix> matrices);
    };

    /**
     * @brief 读取 *.mat 中的矩阵流；Read matrix stream from *.mat.
     * @param path 输入路径；Input path.
     * @return 矩阵序列；Matrix sequence.
     */
    [[nodiscard]] std::vector<Matrix> read_mat_file(const std::filesystem::path &path);

    /**
     * @brief 写入 *.mat 矩阵流；Write matrix stream to *.mat.
     * @param path 输出路径；Output path.
     * @param matrices 矩阵序列；Matrix sequence.
     */
    void write_mat_file(const std::filesystem::path &path, std::span<const Matrix> matrices);

    /**
     * @brief 读取文本矩阵流；Read matrix stream from text file.
     * @param path 输入路径；Input path.
     * @return 矩阵序列；Matrix sequence.
     */
    [[nodiscard]] std::vector<Matrix> read_txt_file(const std::filesystem::path &path);

    /**
     * @brief 写入文本矩阵流；Write matrix stream to text file.
     * @param path 输出路径；Output path.
     * @param matrices 矩阵序列；Matrix sequence.
     */
    void write_txt_file(const std::filesystem::path &path, std::span<const Matrix> matrices);

    /**
     * @brief 类型别名：*.mat 输入流；Type alias: *.mat input stream.
     */
    using MatInputStream = MatrixInputStream<MatFilePolicy>;

    /**
     * @brief 类型别名：*.mat 输出流；Type alias: *.mat output stream.
     */
    using MatOutputStream = MatrixOutputStream<MatFilePolicy>;

    /**
     * @brief 类型别名：*.txt 输入流；Type alias: *.txt input stream.
     */
    using TxtInputStream = MatrixInputStream<TxtFilePolicy>;

    /**
     * @brief 类型别名：*.txt 输出流；Type alias: *.txt output stream.
     */
    using TxtOutputStream = MatrixOutputStream<TxtFilePolicy>;
} // namespace jacobi::svd::io
