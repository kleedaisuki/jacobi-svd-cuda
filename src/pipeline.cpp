#include "pipeline.hpp"

#include "io.hpp"

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>

namespace jacobi::svd::pipeline
{
    namespace
    {
        /**
         * @brief 计算 rows*columns（带溢出检查）；Compute rows*columns with overflow check.
         * @param rows 行数；Row count.
         * @param columns 列数；Column count.
         * @return 元素数量；Element count.
         */
        [[nodiscard]] std::size_t checked_element_count(std::size_t rows, std::size_t columns)
        {
            if (rows == 0 || columns == 0)
            {
                return 0;
            }
            if (rows > (std::numeric_limits<std::size_t>::max() / columns))
            {
                throw std::overflow_error("Matrix element count overflow.");
            }
            return rows * columns;
        }

        /**
         * @brief 校验矩阵布局一致性；Validate matrix layout consistency.
         * @param matrix 输入矩阵；Input matrix.
         * @param testcase_index 测试用例索引；Testcase index.
         */
        void validate_testcase_matrix(const io::Matrix &matrix, std::size_t testcase_index)
        {
            if (matrix.rows == 0 || matrix.columns == 0)
            {
                throw std::invalid_argument("Testcase[" + std::to_string(testcase_index) +
                                            "] has zero dimension.");
            }

            const std::size_t expected = checked_element_count(matrix.rows, matrix.columns);
            if (matrix.values.size() != expected)
            {
                throw std::invalid_argument("Testcase[" + std::to_string(testcase_index) +
                                            "] payload size does not match rows*columns.");
            }
        }

        /**
         * @brief 解析文件格式（支持 auto）；Resolve file format with auto-detection.
         * @param requested 请求格式；Requested format.
         * @param path 文件路径；File path.
         * @return 实际格式；Resolved format.
         */
        [[nodiscard]] MatrixFileFormat resolve_file_format(MatrixFileFormat requested,
                                                           const std::filesystem::path &path)
        {
            if (requested != MatrixFileFormat::auto_detect)
            {
                return requested;
            }

            std::string extension = path.extension().string();
            std::transform(extension.begin(),
                           extension.end(),
                           extension.begin(),
                           [](unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });

            if (extension == ".mat")
            {
                return MatrixFileFormat::mat;
            }
            if (extension == ".txt")
            {
                return MatrixFileFormat::txt;
            }

            throw std::invalid_argument("Cannot auto-detect file format for path: " + path.string());
        }

        /**
         * @brief 确保输出目录存在；Ensure output directory exists.
         * @param output_path 输出文件路径；Output file path.
         */
        void ensure_output_directory(const std::filesystem::path &output_path)
        {
            const std::filesystem::path parent = output_path.parent_path();
            if (!parent.empty())
            {
                std::filesystem::create_directories(parent);
            }
        }

        /**
         * @brief 输出数据包（单个 testcase 的 U/Sigma/V）；Output packet containing U/Sigma/V of one testcase.
         */
        struct OutputPacket final
        {
            /**
             * @brief 测试用例索引；Testcase index.
             */
            std::size_t testcase_index = 0;

            /**
             * @brief 当前用例 sweep 次数；Sweep count of this testcase.
             */
            int sweeps = 0;

            /**
             * @brief 左奇异矩阵 U；Left singular matrix U.
             */
            io::Matrix u;

            /**
             * @brief 奇异值矩阵 Sigma(1xn)；Singular value matrix Sigma(1xn).
             */
            io::Matrix sigma;

            /**
             * @brief 右奇异矩阵 V；Right singular matrix V.
             */
            io::Matrix v;
        };

        /**
         * @brief 测试用例输入阶段；Input stage for testcase stream.
         */
        class TestcaseSource final
        {
        public:
            /**
             * @brief 构造输入阶段对象；Construct input-stage object.
             * @param input_path 输入路径；Input path.
             * @param format 输入格式；Input format.
             */
            TestcaseSource(const std::filesystem::path &input_path, MatrixFileFormat format)
                : stream_(build_stream(input_path, format))
            {
            }

            /**
             * @brief 读取下一张测试矩阵；Read next testcase matrix.
             * @param matrix 输出矩阵；Output matrix.
             * @return 成功返回 true；Returns true on success.
             */
            bool read_next(io::Matrix &matrix)
            {
                return std::visit([&matrix](auto &stream) {
                    return stream.read_one(matrix);
                },
                                  stream_);
            }

        private:
            /**
             * @brief 输入流变体类型；Input stream variant type.
             */
            using InputStreamVariant = std::variant<io::MatInputStream, io::TxtInputStream>;

            /**
             * @brief 构造输入流；Build input stream.
             * @param input_path 输入路径；Input path.
             * @param format 输入格式；Input format.
             * @return 输入流变体；Input stream variant.
             */
            [[nodiscard]] static InputStreamVariant build_stream(const std::filesystem::path &input_path,
                                                                 MatrixFileFormat format)
            {
                if (format == MatrixFileFormat::mat)
                {
                    return io::MatInputStream(input_path);
                }
                if (format == MatrixFileFormat::txt)
                {
                    return io::TxtInputStream(input_path);
                }

                throw std::invalid_argument("Unsupported testcase input format.");
            }

            /**
             * @brief 输入流实例；Input stream instance.
             */
            InputStreamVariant stream_;
        };

        /**
         * @brief 核函数执行阶段；Kernel execution stage.
         */
        class KernelStage final
        {
        public:
            /**
             * @brief 构造核函数阶段对象；Construct kernel-stage object.
             * @param config 核函数配置；Kernel configuration.
             */
            explicit KernelStage(JacobiSvdConfig config)
                : config_(config)
            {
            }

            /**
             * @brief 执行一次 testcase 的 SVD；Run one testcase SVD.
             * @param testcase 输入矩阵；Input matrix.
             * @param testcase_index 测试用例索引；Testcase index.
             * @return 输出数据包；Output packet.
             */
            [[nodiscard]] OutputPacket execute(const io::Matrix &testcase, std::size_t testcase_index) const
            {
                validate_testcase_matrix(testcase, testcase_index);

                JacobiSvdResult result = one_sided_jacobi_svd(testcase.values,
                                                              testcase.rows,
                                                              testcase.columns,
                                                              config_);

                OutputPacket packet;
                packet.testcase_index = testcase_index;
                packet.sweeps = result.sweeps;
                packet.u = io::Matrix{
                    .rows = result.rows,
                    .columns = result.columns,
                    .values = std::move(result.u),
                };
                packet.sigma = io::Matrix{
                    .rows = 1,
                    .columns = result.columns,
                    .values = std::move(result.sigma),
                };
                packet.v = io::Matrix{
                    .rows = result.columns,
                    .columns = result.columns,
                    .values = std::move(result.v),
                };
                return packet;
            }

        private:
            /**
             * @brief 核函数配置；Kernel configuration.
             */
            JacobiSvdConfig config_{};
        };

        /**
         * @brief 结果写出器（单线程消费者）；Result writer (single consumer thread).
         */
        class ResultWriter final
        {
        public:
            /**
             * @brief 构造结果写出器；Construct result writer.
             * @param output_path 输出路径；Output path.
             * @param format 输出格式；Output format.
             */
            ResultWriter(const std::filesystem::path &output_path, MatrixFileFormat format)
                : stream_(build_stream(output_path, format))
            {
            }

            /**
             * @brief 写出一条输出数据包；Write one output packet.
             * @param packet 输出数据包；Output packet.
             */
            void write_packet(const OutputPacket &packet)
            {
                write_matrix(packet.u);
                write_matrix(packet.sigma);
                write_matrix(packet.v);
            }

            /**
             * @brief 刷新输出流；Flush output stream.
             */
            void flush()
            {
                std::visit([](auto &stream) {
                    stream.flush();
                },
                           stream_);
            }

        private:
            /**
             * @brief 输出流变体类型；Output stream variant type.
             */
            using OutputStreamVariant = std::variant<io::MatOutputStream, io::TxtOutputStream>;

            /**
             * @brief 构造输出流；Build output stream.
             * @param output_path 输出路径；Output path.
             * @param format 输出格式；Output format.
             * @return 输出流变体；Output stream variant.
             */
            [[nodiscard]] static OutputStreamVariant build_stream(const std::filesystem::path &output_path,
                                                                  MatrixFileFormat format)
            {
                if (format == MatrixFileFormat::mat)
                {
                    return io::MatOutputStream(output_path);
                }
                if (format == MatrixFileFormat::txt)
                {
                    return io::TxtOutputStream(output_path);
                }

                throw std::invalid_argument("Unsupported pipeline output format.");
            }

            /**
             * @brief 写出单张矩阵；Write one matrix.
             * @param matrix 输入矩阵；Input matrix.
             */
            void write_matrix(const io::Matrix &matrix)
            {
                std::visit([&matrix](auto &stream) {
                    stream.write_one(matrix);
                },
                           stream_);
            }

            /**
             * @brief 输出流实例；Output stream instance.
             */
            OutputStreamVariant stream_;
        };

        /**
         * @brief 异步输出阶段（生产者-消费者）；Asynchronous output stage (producer-consumer).
         * @note 生产者在主线程执行 kernel，消费者线程负责落盘；Producer runs kernels on main thread while consumer thread persists results.
         */
        class AsyncOutputStage final
        {
        public:
            /**
             * @brief 构造异步输出阶段；Construct asynchronous output stage.
             * @param output_path 输出路径；Output path.
             * @param format 输出格式；Output format.
             * @param queue_capacity 队列容量；Queue capacity.
             */
            AsyncOutputStage(const std::filesystem::path &output_path,
                             MatrixFileFormat format,
                             std::size_t queue_capacity)
                : queue_capacity_(std::max<std::size_t>(queue_capacity, 1)),
                  consumer_thread_(&AsyncOutputStage::consumer_main, this, output_path, format)
            {
            }

            /**
             * @brief 析构时安全关闭线程；Safely close thread on destruction.
             */
            ~AsyncOutputStage()
            {
                close_noexcept();
            }

            /**
             * @brief 禁止拷贝构造；Copy construction is disabled.
             */
            AsyncOutputStage(const AsyncOutputStage &) = delete;

            /**
             * @brief 禁止拷贝赋值；Copy assignment is disabled.
             * @return 当前对象引用；Reference to current object.
             */
            AsyncOutputStage &operator=(const AsyncOutputStage &) = delete;

            /**
             * @brief 提交输出数据包；Submit one output packet.
             * @param packet 输出数据包；Output packet.
             */
            void submit(OutputPacket packet)
            {
                std::unique_lock<std::mutex> lock(mutex_);
                producer_cv_.wait(lock, [this] {
                    return queue_.size() < queue_capacity_ || closed_ || worker_error_ != nullptr;
                });
                rethrow_worker_error_locked();
                if (closed_)
                {
                    throw std::runtime_error("Output stage is closed.");
                }
                queue_.push(std::move(packet));
                consumer_cv_.notify_one();
            }

            /**
             * @brief 关闭输出阶段并等待线程结束；Close output stage and wait for thread termination.
             */
            void close()
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    closed_ = true;
                }

                consumer_cv_.notify_all();
                producer_cv_.notify_all();

                if (consumer_thread_.joinable())
                {
                    consumer_thread_.join();
                }

                if (worker_error_ != nullptr)
                {
                    std::rethrow_exception(worker_error_);
                }
            }

        private:
            /**
             * @brief 消费者主循环；Consumer main loop.
             * @param output_path 输出路径；Output path.
             * @param format 输出格式；Output format.
             */
            void consumer_main(const std::filesystem::path output_path, MatrixFileFormat format)
            {
                try
                {
                    ensure_output_directory(output_path);
                    ResultWriter writer(output_path, format);

                    for (;;)
                    {
                        OutputPacket packet;
                        {
                            std::unique_lock<std::mutex> lock(mutex_);
                            consumer_cv_.wait(lock, [this] {
                                return !queue_.empty() || closed_;
                            });

                            if (queue_.empty())
                            {
                                break;
                            }

                            packet = std::move(queue_.front());
                            queue_.pop();
                            producer_cv_.notify_one();
                        }
                        writer.write_packet(packet);
                    }

                    writer.flush();
                }
                catch (...)
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (worker_error_ == nullptr)
                    {
                        worker_error_ = std::current_exception();
                    }
                    closed_ = true;
                    producer_cv_.notify_all();
                    consumer_cv_.notify_all();
                }
            }

            /**
             * @brief 在锁内抛出消费者异常；Rethrow consumer exception while lock is held.
             */
            void rethrow_worker_error_locked() const
            {
                if (worker_error_ != nullptr)
                {
                    std::rethrow_exception(worker_error_);
                }
            }

            /**
             * @brief noexcept 关闭辅助函数；noexcept close helper.
             */
            void close_noexcept() noexcept
            {
                try
                {
                    close();
                }
                catch (...)
                {
                }
            }

            /**
             * @brief 队列容量；Queue capacity.
             */
            std::size_t queue_capacity_ = 1;

            /**
             * @brief 生产者-消费者队列；Producer-consumer queue.
             */
            std::queue<OutputPacket> queue_;

            /**
             * @brief 互斥锁；Mutex.
             */
            std::mutex mutex_;

            /**
             * @brief 消费者条件变量；Consumer condition variable.
             */
            std::condition_variable consumer_cv_;

            /**
             * @brief 生产者条件变量；Producer condition variable.
             */
            std::condition_variable producer_cv_;

            /**
             * @brief 是否已请求关闭；Whether close is requested.
             */
            bool closed_ = false;

            /**
             * @brief 消费者线程异常；Consumer thread exception.
             */
            std::exception_ptr worker_error_;

            /**
             * @brief 消费者线程；Consumer thread.
             */
            std::thread consumer_thread_;
        };
    } // namespace

    JacobiSvdPipeline::JacobiSvdPipeline(PipelineConfig config)
        : config_(std::move(config))
    {
    }

    PipelineReport JacobiSvdPipeline::run() const
    {
        if (config_.input_path.empty())
        {
            throw std::invalid_argument("Pipeline input path is empty.");
        }
        if (config_.output_path.empty())
        {
            throw std::invalid_argument("Pipeline output path is empty.");
        }

        const MatrixFileFormat input_format = resolve_file_format(config_.input_format, config_.input_path);
        const MatrixFileFormat output_format = resolve_file_format(config_.output_format, config_.output_path);

        TestcaseSource source(config_.input_path, input_format);
        KernelStage kernel(config_.kernel_config);
        AsyncOutputStage output(config_.output_path, output_format, config_.max_queued_results);

        PipelineReport report;
        std::size_t testcase_index = 0;

        io::Matrix testcase;
        while (source.read_next(testcase))
        {
            OutputPacket packet = kernel.execute(testcase, testcase_index);
            report.testcase_count += 1;
            report.emitted_matrix_count += 3;
            report.total_sweeps += static_cast<std::size_t>(packet.sweeps);
            output.submit(std::move(packet));
            ++testcase_index;
        }

        output.close();
        return report;
    }

    PipelineReport run_pipeline(const PipelineConfig &config)
    {
        return JacobiSvdPipeline(config).run();
    }
} // namespace jacobi::svd::pipeline

