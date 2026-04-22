#pragma once

#include <cstddef>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
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
     * @brief 页锁定主机缓冲区；Pinned host buffer with one contiguous allocation.
     * @note 该缓冲区通过 cudaMallocHost/cudaFreeHost 管理；This buffer is managed by cudaMallocHost/cudaFreeHost.
     */
    class PinnedHostTaskBuffer final
    {
    public:
        /**
         * @brief 构造空缓冲；Construct an empty buffer.
         */
        PinnedHostTaskBuffer() = default;

        /**
         * @brief 析构并释放缓冲；Destroy and release the buffer.
         */
        ~PinnedHostTaskBuffer();

        /**
         * @brief 禁止拷贝构造；Copy construction is disabled.
         */
        PinnedHostTaskBuffer(const PinnedHostTaskBuffer &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        PinnedHostTaskBuffer &operator=(const PinnedHostTaskBuffer &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        PinnedHostTaskBuffer(PinnedHostTaskBuffer &&other) noexcept;

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        PinnedHostTaskBuffer &operator=(PinnedHostTaskBuffer &&other) noexcept;

        /**
         * @brief 预留输入区与工作区；Reserve one block for input and workspace.
         * @param input_bytes 输入区字节数；Input byte size.
         * @param workspace_bytes 工作区字节数；Workspace byte size.
         */
        void reserve(std::size_t input_bytes, std::size_t workspace_bytes);

        /**
         * @brief 获取可写输入区；Get mutable input region.
         * @return 输入区字节视图；Input byte span.
         */
        [[nodiscard]] std::span<std::byte> mutable_input_bytes() noexcept;

        /**
         * @brief 获取只读输入区；Get const input region.
         * @return 输入区字节视图；Input byte span.
         */
        [[nodiscard]] std::span<const std::byte> input_bytes() const noexcept;

        /**
         * @brief 获取可写工作区；Get mutable workspace region.
         * @return 工作区字节视图；Workspace byte span.
         */
        [[nodiscard]] std::span<std::byte> mutable_workspace_bytes() noexcept;

        /**
         * @brief 获取只读工作区；Get const workspace region.
         * @return 工作区字节视图；Workspace byte span.
         */
        [[nodiscard]] std::span<const std::byte> workspace_bytes() const noexcept;

        /**
         * @brief 当前容量（字节）；Current allocated capacity in bytes.
         * @return 容量；Capacity.
         */
        [[nodiscard]] std::size_t capacity_bytes() const noexcept;

        /**
         * @brief 当前输入区大小（字节）；Current input size in bytes.
         * @return 输入区大小；Input size.
         */
        [[nodiscard]] std::size_t input_size_bytes() const noexcept;

        /**
         * @brief 当前工作区大小（字节）；Current workspace size in bytes.
         * @return 工作区大小；Workspace size.
         */
        [[nodiscard]] std::size_t workspace_size_bytes() const noexcept;

    private:
        /**
         * @brief 释放底层缓冲；Release underlying allocation.
         */
        void release() noexcept;

        /**
         * @brief 从另一个对象移动资源；Move resources from another object.
         * @param other 源对象；Source object.
         */
        void move_from(PinnedHostTaskBuffer &&other) noexcept;

        /**
         * @brief 输入区起始地址；Input region base address.
         */
        std::byte *data_ = nullptr;

        /**
         * @brief 当前容量（字节）；Current capacity in bytes.
         */
        std::size_t capacity_bytes_ = 0;

        /**
         * @brief 输入区大小（字节）；Input size in bytes.
         */
        std::size_t input_size_bytes_ = 0;

        /**
         * @brief 工作区大小（字节）；Workspace size in bytes.
         */
        std::size_t workspace_size_bytes_ = 0;
    };

    /**
     * @brief 单次派发任务；One dispatch task for a single *.mat matrix record.
     * @note 输入区保存原始网络字节序 payload，工作区用于后续解析/计算；Input region stores raw network-order payload, workspace is reserved for later decode/compute.
     */
    struct MatDispatchTask final
    {
        /**
         * @brief 派发序号；Dispatch sequence index.
         */
        std::size_t sequence_index = 0;

        /**
         * @brief 矩阵行数；Matrix row count.
         */
        std::size_t rows = 0;

        /**
         * @brief 矩阵列数；Matrix column count.
         */
        std::size_t columns = 0;

        /**
         * @brief 单块页锁定缓冲；Single pinned block containing input+workspace.
         */
        PinnedHostTaskBuffer buffer;
    };

    /**
     * @brief *.mat 单游标派发读取器；Single-cursor dispatch reader for *.mat.
     * @note 该类为栈对象设计，不使用 pImpl（pointer to implementation）；This class is stack-allocated and does not use pImpl.
     */
    class MatDispatchReader final
    {
    public:
        /**
         * @brief 通过路径构造派发读取器；Construct dispatch reader from file path.
         * @param path 输入路径；Input path.
         */
        explicit MatDispatchReader(const std::filesystem::path &path);

        /**
         * @brief 读取并填充下一条任务；Read and populate next dispatch task.
         * @param task 输出任务（可移动复用）；Output task (movable and reusable).
         * @param workspace_bytes 预留工作区字节数；Reserved workspace byte size.
         * @return 成功读取返回 true，EOF 返回 false；Returns true if one task is read, false on EOF.
         */
        [[nodiscard]] bool read_next(MatDispatchTask &task, std::size_t workspace_bytes = 0);

    private:
        /**
         * @brief 输入文件流；Input file stream.
         */
        std::ifstream input_;

        /**
         * @brief 下一个派发序号；Next dispatch sequence index.
         */
        std::size_t next_sequence_index_ = 0;
    };

    /**
     * @brief 将派发任务解析为矩阵；Decode dispatch task payload into Matrix.
     * @param task 派发任务；Dispatch task.
     * @return 解码后的矩阵；Decoded matrix.
     */
    [[nodiscard]] Matrix decode_dispatch_task_matrix(const MatDispatchTask &task);

    /**
     * @brief *.mat 读取器前置声明；Forward declaration of *.mat reader.
     */
    class MatReader;

    /**
     * @brief *.mat 写入器前置声明；Forward declaration of *.mat writer.
     */
    class MatWriter;

    /**
     * @brief *.txt 读取器前置声明；Forward declaration of *.txt reader.
     */
    class TxtReader;

    /**
     * @brief *.txt 写入器前置声明；Forward declaration of *.txt writer.
     */
    class TxtWriter;

    /**
     * @brief *.mat 文件 policy；Policy for *.mat files.
     */
    struct MatFilePolicy final
    {
        /**
         * @brief 输入状态类型；Input state type.
         */
        using Reader = MatReader;

        /**
         * @brief 输出状态类型；Output state type.
         */
        using Writer = MatWriter;

        /**
         * @brief 打开 *.mat 读取器；Open *.mat reader.
         * @param path 输入路径；Input path.
         * @return 读取器对象；Reader object.
         */
        [[nodiscard]] static Reader open_reader(const std::filesystem::path &path);

        /**
         * @brief 打开 *.mat 写入器；Open *.mat writer.
         * @param path 输出路径；Output path.
         * @return 写入器对象；Writer object.
         */
        [[nodiscard]] static Writer open_writer(const std::filesystem::path &path);

        /**
         * @brief 读取下一张矩阵；Read next matrix.
         * @param reader 读取器；Reader.
         * @param matrix 输出矩阵；Output matrix.
         * @return 成功读取返回 true，EOF 返回 false；Returns true if one matrix is read, false on EOF.
         */
        [[nodiscard]] static bool read_next(Reader &reader, Matrix &matrix);

        /**
         * @brief 写入下一张矩阵；Write next matrix.
         * @param writer 写入器；Writer.
         * @param matrix 输入矩阵；Input matrix.
         */
        static void write_next(Writer &writer, const Matrix &matrix);

        /**
         * @brief 刷新写入缓冲；Flush output state.
         * @param writer 写入器；Writer.
         */
        static void flush(Writer &writer);

        /**
         * @brief 批量读取（兼容接口）；Bulk read (compatibility API).
         * @param path 输入路径；Input path.
         * @return 矩阵序列；Matrix sequence.
         */
        [[nodiscard]] static std::vector<Matrix> read(const std::filesystem::path &path);

        /**
         * @brief 批量写入（兼容接口）；Bulk write (compatibility API).
         * @param path 输出路径；Output path.
         * @param matrices 矩阵序列；Matrix sequence.
         */
        static void write(const std::filesystem::path &path, std::span<const Matrix> matrices);
    };

    /**
     * @brief 文本矩阵文件 policy；Policy for text matrix files.
     * @note 行内空格分隔，行间换行分隔，矩阵之间空行分隔；Values are space-separated, rows are newline-separated, matrices are separated by blank lines.
     */
    struct TxtFilePolicy final
    {
        /**
         * @brief 输入状态类型；Input state type.
         */
        using Reader = TxtReader;

        /**
         * @brief 输出状态类型；Output state type.
         */
        using Writer = TxtWriter;

        /**
         * @brief 打开文本读取器；Open text reader.
         * @param path 输入路径；Input path.
         * @return 读取器对象；Reader object.
         */
        [[nodiscard]] static Reader open_reader(const std::filesystem::path &path);

        /**
         * @brief 打开文本写入器；Open text writer.
         * @param path 输出路径；Output path.
         * @return 写入器对象；Writer object.
         */
        [[nodiscard]] static Writer open_writer(const std::filesystem::path &path);

        /**
         * @brief 读取下一张矩阵；Read next matrix.
         * @param reader 读取器；Reader.
         * @param matrix 输出矩阵；Output matrix.
         * @return 成功读取返回 true，EOF 返回 false；Returns true if one matrix is read, false on EOF.
         */
        [[nodiscard]] static bool read_next(Reader &reader, Matrix &matrix);

        /**
         * @brief 写入下一张矩阵；Write next matrix.
         * @param writer 写入器；Writer.
         * @param matrix 输入矩阵；Input matrix.
         */
        static void write_next(Writer &writer, const Matrix &matrix);

        /**
         * @brief 刷新写入缓冲；Flush output state.
         * @param writer 写入器；Writer.
         */
        static void flush(Writer &writer);

        /**
         * @brief 批量读取（兼容接口）；Bulk read (compatibility API).
         * @param path 输入路径；Input path.
         * @return 矩阵序列；Matrix sequence.
         */
        [[nodiscard]] static std::vector<Matrix> read(const std::filesystem::path &path);

        /**
         * @brief 批量写入（兼容接口）；Bulk write (compatibility API).
         * @param path 输出路径；Output path.
         * @param matrices 矩阵序列；Matrix sequence.
         */
        static void write(const std::filesystem::path &path, std::span<const Matrix> matrices);
    };

    /**
     * @brief *.mat 读取器实现包装；Implementation wrapper of *.mat reader.
     */
    class MatReader final
    {
    public:
        /**
         * @brief 通过路径构造读取器；Construct reader from path.
         * @param path 输入路径；Input path.
         */
        explicit MatReader(const std::filesystem::path &path);

        /**
         * @brief 析构读取器；Destroy reader.
         */
        ~MatReader();

        /**
         * @brief 禁止拷贝构造；Copy constructor is disabled.
         */
        MatReader(const MatReader &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        MatReader &operator=(const MatReader &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        MatReader(MatReader &&other) noexcept;

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        MatReader &operator=(MatReader &&other) noexcept;

    private:
        /**
         * @brief 实现体前置声明；Forward declaration of implementation.
         */
        struct Impl;

        /**
         * @brief 唯一实现体指针；Unique pointer of implementation.
         */
        std::unique_ptr<Impl> impl_;

        /**
         * @brief 授权 policy 访问实现体；Grant policy access to implementation.
         */
        friend struct MatFilePolicy;
    };

    /**
     * @brief *.mat 写入器实现包装；Implementation wrapper of *.mat writer.
     */
    class MatWriter final
    {
    public:
        /**
         * @brief 通过路径构造写入器；Construct writer from path.
         * @param path 输出路径；Output path.
         */
        explicit MatWriter(const std::filesystem::path &path);

        /**
         * @brief 析构写入器；Destroy writer.
         */
        ~MatWriter();

        /**
         * @brief 禁止拷贝构造；Copy constructor is disabled.
         */
        MatWriter(const MatWriter &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        MatWriter &operator=(const MatWriter &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        MatWriter(MatWriter &&other) noexcept;

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        MatWriter &operator=(MatWriter &&other) noexcept;

    private:
        /**
         * @brief 实现体前置声明；Forward declaration of implementation.
         */
        struct Impl;

        /**
         * @brief 唯一实现体指针；Unique pointer of implementation.
         */
        std::unique_ptr<Impl> impl_;

        /**
         * @brief 授权 policy 访问实现体；Grant policy access to implementation.
         */
        friend struct MatFilePolicy;
    };

    /**
     * @brief *.txt 读取器实现包装；Implementation wrapper of *.txt reader.
     */
    class TxtReader final
    {
    public:
        /**
         * @brief 通过路径构造读取器；Construct reader from path.
         * @param path 输入路径；Input path.
         */
        explicit TxtReader(const std::filesystem::path &path);

        /**
         * @brief 析构读取器；Destroy reader.
         */
        ~TxtReader();

        /**
         * @brief 禁止拷贝构造；Copy constructor is disabled.
         */
        TxtReader(const TxtReader &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        TxtReader &operator=(const TxtReader &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        TxtReader(TxtReader &&other) noexcept;

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        TxtReader &operator=(TxtReader &&other) noexcept;

    private:
        /**
         * @brief 实现体前置声明；Forward declaration of implementation.
         */
        struct Impl;

        /**
         * @brief 唯一实现体指针；Unique pointer of implementation.
         */
        std::unique_ptr<Impl> impl_;

        /**
         * @brief 授权 policy 访问实现体；Grant policy access to implementation.
         */
        friend struct TxtFilePolicy;
    };

    /**
     * @brief *.txt 写入器实现包装；Implementation wrapper of *.txt writer.
     */
    class TxtWriter final
    {
    public:
        /**
         * @brief 通过路径构造写入器；Construct writer from path.
         * @param path 输出路径；Output path.
         */
        explicit TxtWriter(const std::filesystem::path &path);

        /**
         * @brief 析构写入器；Destroy writer.
         */
        ~TxtWriter();

        /**
         * @brief 禁止拷贝构造；Copy constructor is disabled.
         */
        TxtWriter(const TxtWriter &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        TxtWriter &operator=(const TxtWriter &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        TxtWriter(TxtWriter &&other) noexcept;

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        TxtWriter &operator=(TxtWriter &&other) noexcept;

    private:
        /**
         * @brief 实现体前置声明；Forward declaration of implementation.
         */
        struct Impl;

        /**
         * @brief 唯一实现体指针；Unique pointer of implementation.
         */
        std::unique_ptr<Impl> impl_;

        /**
         * @brief 授权 policy 访问实现体；Grant policy access to implementation.
         */
        friend struct TxtFilePolicy;
    };

    /**
     * @brief 矩阵输入 policy 概念；Concept for matrix input policy.
     * @tparam Policy policy 类型；Policy type.
     */
    template <typename Policy>
    concept MatrixInputPolicy = requires(const std::filesystem::path &path, typename Policy::Reader &reader, Matrix &matrix)
    {
        typename Policy::Reader;
        {
            Policy::open_reader(path)
        } -> std::same_as<typename Policy::Reader>;
        {
            Policy::read_next(reader, matrix)
        } -> std::same_as<bool>;
    };

    /**
     * @brief 矩阵输出 policy 概念；Concept for matrix output policy.
     * @tparam Policy policy 类型；Policy type.
     */
    template <typename Policy>
    concept MatrixOutputPolicy = requires(const std::filesystem::path &path,
                                          typename Policy::Writer &writer,
                                          const Matrix &matrix)
    {
        typename Policy::Writer;
        {
            Policy::open_writer(path)
        } -> std::same_as<typename Policy::Writer>;
        {
            Policy::write_next(writer, matrix)
        } -> std::same_as<void>;
        {
            Policy::flush(writer)
        } -> std::same_as<void>;
    };

    /**
     * @brief 矩阵输入流模板；Matrix input stream template.
     * @tparam Policy 输入 policy；Input policy.
     */
    template <MatrixInputPolicy Policy>
    class MatrixInputStream final
    {
    public:
        /**
         * @brief 构造输入流；Construct input stream.
         * @param path 文件路径；File path.
         */
        explicit MatrixInputStream(const std::filesystem::path &path)
            : reader_(Policy::open_reader(path))
        {
        }

        /**
         * @brief 读取下一张矩阵；Read one matrix.
         * @param matrix 输出矩阵；Output matrix.
         * @return 成功读取返回 true，EOF 返回 false；Returns true if one matrix is read, false on EOF.
         */
        bool read_one(Matrix &matrix)
        {
            if (eof_)
            {
                return false;
            }

            const bool has_matrix = Policy::read_next(reader_, matrix);
            eof_ = !has_matrix;
            return has_matrix;
        }

        /**
         * @brief 操作符重载：读取下一张矩阵；Operator overload: read one matrix.
         * @param matrix 输出矩阵；Output matrix.
         * @return 当前输入流对象；Current input stream object.
         */
        MatrixInputStream &operator>>(Matrix &matrix)
        {
            (void)read_one(matrix);
            return *this;
        }

        /**
         * @brief 是否已到文件末尾；Whether EOF is reached.
         * @return EOF 状态；EOF state.
         */
        [[nodiscard]] bool eof() const noexcept
        {
            return eof_;
        }

        /**
         * @brief 读取全部矩阵（兼容接口）；Read all matrices (compatibility API).
         * @return 矩阵序列；Matrix sequence.
         */
        [[nodiscard]] std::vector<Matrix> read_all()
        {
            std::vector<Matrix> matrices;
            Matrix matrix;
            while (read_one(matrix))
            {
                matrices.push_back(std::move(matrix));
            }
            return matrices;
        }

        /**
         * @brief 布尔语义：尚未 EOF；Boolean semantics: not EOF yet.
         * @return true 表示可继续尝试读取；true means stream can still be read.
         */
        explicit operator bool() const noexcept
        {
            return !eof_;
        }

    private:
        /**
         * @brief 读取器状态；Reader state.
         */
        typename Policy::Reader reader_;

        /**
         * @brief EOF 状态；EOF state.
         */
        bool eof_ = false;
    };

    /**
     * @brief 矩阵输出流模板；Matrix output stream template.
     * @tparam Policy 输出 policy；Output policy.
     */
    template <MatrixOutputPolicy Policy>
    class MatrixOutputStream final
    {
    public:
        /**
         * @brief 构造输出流；Construct output stream.
         * @param path 文件路径；File path.
         */
        explicit MatrixOutputStream(const std::filesystem::path &path)
            : writer_(Policy::open_writer(path))
        {
        }

        /**
         * @brief 写入一张矩阵；Write one matrix.
         * @param matrix 输入矩阵；Input matrix.
         */
        void write_one(const Matrix &matrix)
        {
            Policy::write_next(writer_, matrix);
        }

        /**
         * @brief 操作符重载：写入一张矩阵；Operator overload: write one matrix.
         * @param matrix 输入矩阵；Input matrix.
         * @return 当前输出流对象；Current output stream object.
         */
        MatrixOutputStream &operator<<(const Matrix &matrix)
        {
            write_one(matrix);
            return *this;
        }

        /**
         * @brief 刷新输出；Flush output.
         */
        void flush()
        {
            Policy::flush(writer_);
        }

        /**
         * @brief 写入全部矩阵（兼容接口）；Write all matrices (compatibility API).
         * @param matrices 矩阵序列；Matrix sequence.
         */
        void write_all(std::span<const Matrix> matrices)
        {
            for (const Matrix &matrix : matrices)
            {
                write_one(matrix);
            }
            flush();
        }

    private:
        /**
         * @brief 写入器状态；Writer state.
         */
        typename Policy::Writer writer_;
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
