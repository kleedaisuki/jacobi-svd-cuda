#include "pipeline.hpp"

#include "io.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

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
         * @brief 输出数据包（单个 testcase 的 U/Sigma/V）；Output packet for one testcase (U/Sigma/V).
         */
        struct OutputPacket final
        {
            /**
             * @brief 测试用例索引；Testcase index.
             */
            std::size_t testcase_index = 0;

            /**
             * @brief 当前用例 sweep 次数；Sweep count for this testcase.
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
         * @brief 结果写出器（单线程）；Result writer (single thread).
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
             * @brief 写出一个数据包；Write one output packet.
             * @param packet 输出数据包；Output packet.
             */
            void write_packet(const OutputPacket &packet)
            {
                write_matrix(packet.u);
                write_matrix(packet.sigma);
                write_matrix(packet.v);
            }

            /**
             * @brief 刷新输出；Flush output stream.
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
         * @brief 全局线程池（Thread Pool）实现；Implementation of global thread pool.
         */
        class GlobalThreadPool final
        {
        public:
            /**
             * @brief 构造线程池；Construct thread pool.
             * @param worker_count 工作线程数；Worker thread count.
             */
            explicit GlobalThreadPool(std::size_t worker_count)
                : worker_count_(std::max<std::size_t>(worker_count, 1))
            {
                workers_.reserve(worker_count_);
                for (std::size_t index = 0; index < worker_count_; ++index)
                {
                    workers_.emplace_back([this] {
                        worker_main();
                    });
                }
            }

            /**
             * @brief 析构线程池；Destroy thread pool.
             */
            ~GlobalThreadPool()
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    stopping_ = true;
                }
                consumer_cv_.notify_all();

                for (std::thread &worker : workers_)
                {
                    if (worker.joinable())
                    {
                        worker.join();
                    }
                }
            }

            /**
             * @brief 禁止拷贝构造；Copy constructor is disabled.
             */
            GlobalThreadPool(const GlobalThreadPool &) = delete;

            /**
             * @brief 禁止拷贝赋值；Copy assignment is disabled.
             * @return 当前对象引用；Reference to current object.
             */
            GlobalThreadPool &operator=(const GlobalThreadPool &) = delete;

            /**
             * @brief 提交任务并返回 future；Submit task and return future.
             * @tparam Callable 可调用类型；Callable type.
             * @param callable 任务函数；Task callable.
             * @return 任务 future；Task future.
             */
            template <typename Callable>
            [[nodiscard]] auto submit(Callable &&callable) -> std::future<std::invoke_result_t<std::decay_t<Callable>>>
            {
                using ResultType = std::invoke_result_t<std::decay_t<Callable>>;

                auto packaged_task =
                    std::make_shared<std::packaged_task<ResultType()>>(std::forward<Callable>(callable));
                std::future<ResultType> result = packaged_task->get_future();

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (stopping_)
                    {
                        throw std::runtime_error("Thread pool is stopping.");
                    }
                    queue_.emplace([packaged_task] {
                        (*packaged_task)();
                    });
                }

                consumer_cv_.notify_one();
                return result;
            }

        private:
            /**
             * @brief 工作线程主循环；Worker-thread main loop.
             */
            void worker_main()
            {
                for (;;)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        consumer_cv_.wait(lock, [this] {
                            return stopping_ || !queue_.empty();
                        });

                        if (stopping_ && queue_.empty())
                        {
                            return;
                        }

                        task = std::move(queue_.front());
                        queue_.pop();
                    }

                    task();
                }
            }

            /**
             * @brief 工作线程数量；Worker thread count.
             */
            std::size_t worker_count_ = 1;

            /**
             * @brief 任务队列；Task queue.
             */
            std::queue<std::function<void()>> queue_;

            /**
             * @brief 互斥锁；Mutex.
             */
            std::mutex mutex_;

            /**
             * @brief 条件变量；Condition variable.
             */
            std::condition_variable consumer_cv_;

            /**
             * @brief 停止标志；Stop flag.
             */
            bool stopping_ = false;

            /**
             * @brief 工作线程集合；Worker thread collection.
             */
            std::vector<std::thread> workers_;
        };

        /**
         * @brief 获取默认线程池大小；Get default thread-pool size.
         * @return 工作线程数；Worker thread count.
         */
        [[nodiscard]] std::size_t default_thread_pool_size()
        {
            const unsigned int hardware = std::thread::hardware_concurrency();
            if (hardware == 0U)
            {
                return 4;
            }
            return static_cast<std::size_t>(std::max(1U, hardware));
        }

        /**
         * @brief 全局惰性线程池访问器；Accessor for global lazy-initialized thread pool.
         * @return 全局线程池引用；Reference to global thread pool.
         */
        [[nodiscard]] GlobalThreadPool &global_thread_pool()
        {
            static GlobalThreadPool pool(default_thread_pool_size());
            return pool;
        }

        /**
         * @brief 基于 future 队列的异步写出阶段；Asynchronous output stage based on a future queue.
         */
        class FutureQueueOutputStage final
        {
        public:
            /**
             * @brief 构造异步写出阶段；Construct asynchronous output stage.
             * @param output_path 输出路径；Output path.
             * @param format 输出格式；Output format.
             * @param queue_capacity 完成队列容量；Completion queue capacity.
             */
            FutureQueueOutputStage(const std::filesystem::path &output_path,
                                   MatrixFileFormat format,
                                   std::size_t queue_capacity)
                : queue_capacity_(std::max<std::size_t>(queue_capacity, 1)),
                  consumer_thread_(&FutureQueueOutputStage::consumer_main, this, output_path, format)
            {
            }

            /**
             * @brief 析构并安全关闭；Destroy and safely close.
             */
            ~FutureQueueOutputStage()
            {
                close_noexcept();
            }

            /**
             * @brief 禁止拷贝构造；Copy construction is disabled.
             */
            FutureQueueOutputStage(const FutureQueueOutputStage &) = delete;

            /**
             * @brief 禁止拷贝赋值；Copy assignment is disabled.
             * @return 当前对象引用；Reference to current object.
             */
            FutureQueueOutputStage &operator=(const FutureQueueOutputStage &) = delete;

            /**
             * @brief 提交已完成数据包；Submit one completed packet.
             * @param packet 已完成数据包；Completed output packet.
             */
            void submit(std::future<OutputPacket> future_packet)
            {
                std::unique_lock<std::mutex> lock(mutex_);
                producer_cv_.wait(lock, [this] {
                    return future_queue_.size() < queue_capacity_ || closed_ || worker_error_ != nullptr;
                });
                rethrow_worker_error_locked();

                if (closed_)
                {
                    throw std::runtime_error("Output stage is closed.");
                }

                future_queue_.push(std::move(future_packet));
                consumer_cv_.notify_one();
            }

            /**
             * @brief 正常关闭；Close stage in success path.
             * @param expected_count 期望数据包数量；Expected packet count.
             */
            void close_success(std::size_t expected_count)
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    expected_count_ = expected_count;
                    closed_ = true;
                }
                consumer_cv_.notify_all();
                producer_cv_.notify_all();
                join_and_rethrow();

                if (written_count_ != expected_count_)
                {
                    throw std::runtime_error("Output stage finished with missing packets.");
                }
            }

            /**
             * @brief 异常关闭；Close stage in error path.
             * @param error 异常对象；Exception object.
             */
            void close_error(std::exception_ptr error)
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (worker_error_ == nullptr)
                    {
                        worker_error_ = error;
                    }
                    closed_ = true;
                    abort_ = true;
                }
                consumer_cv_.notify_all();
                producer_cv_.notify_all();
                join_and_rethrow();
            }

        private:
            /**
             * @brief 写线程主循环；Writer-thread main loop.
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
                        std::future<OutputPacket> future_packet;
                        bool has_future = false;

                        {
                            std::unique_lock<std::mutex> lock(mutex_);
                            consumer_cv_.wait(lock, [this] {
                                return !future_queue_.empty() || closed_ || abort_;
                            });

                            if (!future_queue_.empty())
                            {
                                future_packet = std::move(future_queue_.front());
                                future_queue_.pop();
                                has_future = true;
                                producer_cv_.notify_one();
                            }
                            else if (closed_ || abort_)
                            {
                                break;
                            }
                        }

                        if (!has_future)
                        {
                            continue;
                        }

                        OutputPacket packet = future_packet.get();
                        writer.write_packet(packet);
                        ++written_count_;
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
                    abort_ = true;
                    producer_cv_.notify_all();
                    consumer_cv_.notify_all();
                }
            }

            /**
             * @brief 连接写线程并抛出异常；Join writer thread and rethrow error if any.
             */
            void join_and_rethrow()
            {
                if (consumer_thread_.joinable())
                {
                    consumer_thread_.join();
                }

                if (worker_error_ != nullptr)
                {
                    std::rethrow_exception(worker_error_);
                }
            }

            /**
             * @brief 在锁内抛出写线程异常；Rethrow writer-thread exception while lock is held.
             */
            void rethrow_worker_error_locked() const
            {
                if (worker_error_ != nullptr)
                {
                    std::rethrow_exception(worker_error_);
                }
            }

            /**
             * @brief noexcept 关闭助手；Noexcept close helper.
             */
            void close_noexcept() noexcept
            {
                try
                {
                    close_error(std::make_exception_ptr(std::runtime_error("Output stage closed by destructor.")));
                }
                catch (...)
                {
                }
            }

            /**
             * @brief 完成队列容量；Completion queue capacity.
             */
            std::size_t queue_capacity_ = 1;

            /**
             * @brief 输出 future 队列；Output future queue.
             */
            std::queue<std::future<OutputPacket>> future_queue_;

            /**
             * @brief 期望总包数；Expected packet count.
             */
            std::size_t expected_count_ = 0;

            /**
             * @brief 已写包数；Written packet count.
             */
            std::size_t written_count_ = 0;

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
             * @brief 关闭标记；Close flag.
             */
            bool closed_ = false;

            /**
             * @brief 中止标记；Abort flag.
             */
            bool abort_ = false;

            /**
             * @brief 写线程异常；Writer-thread exception.
             */
            std::exception_ptr worker_error_;

            /**
             * @brief 消费线程；Consumer thread.
             */
            std::thread consumer_thread_;
        };

        /**
         * @brief 文本输入读取阶段；Text input stage.
         */
        class TextTestcaseSource final
        {
        public:
            /**
             * @brief 构造文本输入阶段；Construct text input stage.
             * @param input_path 输入路径；Input path.
             */
            explicit TextTestcaseSource(const std::filesystem::path &input_path)
                : stream_(io::TxtInputStream(input_path))
            {
            }

            /**
             * @brief 读取下一张矩阵；Read next matrix.
             * @param matrix 输出矩阵；Output matrix.
             * @return 读取成功返回 true，EOF 返回 false；Returns true on success, false on EOF.
             */
            bool read_next(io::Matrix &matrix)
            {
                return stream_.read_one(matrix);
            }

        private:
            /**
             * @brief 文本输入流；Text input stream.
             */
            io::TxtInputStream stream_;
        };

        /**
         * @brief 核函数执行阶段；Kernel execution stage.
         */
        class KernelStage final
        {
        public:
            /**
             * @brief 构造核函数阶段；Construct kernel stage.
             * @param config 核函数配置；Kernel configuration.
             */
            explicit KernelStage(JacobiSvdConfig config)
                : config_(config)
            {
            }

            /**
             * @brief 执行一次 testcase 的 SVD；Execute SVD for one testcase.
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

        JacobiSvdConfig runtime_kernel_config = config_.kernel_config;
        LayoutTransposeAutoTuneReport tuning_report{};
        if (runtime_kernel_config.layout_transpose_auto_tune &&
            runtime_kernel_config.layout_transpose_mode == LayoutTransposeMode::auto_select)
        {
            tuning_report = auto_tune_layout_transpose_threshold(runtime_kernel_config);
            runtime_kernel_config.layout_transpose_min_columns = tuning_report.recommended_min_columns;
            runtime_kernel_config.layout_transpose_min_elements = tuning_report.recommended_min_elements;
            runtime_kernel_config.layout_transpose_auto_tune = false;
        }

        KernelStage kernel(runtime_kernel_config);
        FutureQueueOutputStage output(config_.output_path, output_format, config_.max_queued_results);
        GlobalThreadPool &pool = global_thread_pool();

        std::atomic<std::size_t> total_sweeps{0};
        std::size_t submitted_count = 0;

        try
        {
            if (input_format == MatrixFileFormat::mat)
            {
                io::MatDispatchReader reader(config_.input_path);
                io::MatDispatchTask dispatch_task;

                while (reader.read_next(dispatch_task))
                {
                    io::MatDispatchTask task_for_worker = std::move(dispatch_task);
                    dispatch_task = io::MatDispatchTask{};

                    const std::size_t testcase_index = task_for_worker.sequence_index;
                    std::future<OutputPacket> packet_future = pool.submit([task = std::move(task_for_worker),
                                                                           testcase_index,
                                                                           &kernel,
                                                                           &total_sweeps]() mutable -> OutputPacket {
                        io::Matrix testcase = io::decode_dispatch_task_matrix(task);
                        OutputPacket packet = kernel.execute(testcase, testcase_index);
                        total_sweeps.fetch_add(static_cast<std::size_t>(packet.sweeps), std::memory_order_relaxed);
                        return packet;
                    });
                    output.submit(std::move(packet_future));

                    submitted_count += 1;
                }
            }
            else
            {
                TextTestcaseSource source(config_.input_path);
                io::Matrix testcase;
                std::size_t testcase_index = 0;

                while (source.read_next(testcase))
                {
                    io::Matrix testcase_for_worker = std::move(testcase);
                    testcase = io::Matrix{};

                    std::future<OutputPacket> packet_future = pool.submit([testcase_data = std::move(testcase_for_worker),
                                                                           testcase_index,
                                                                           &kernel,
                                                                           &total_sweeps]() mutable -> OutputPacket {
                        OutputPacket packet = kernel.execute(testcase_data, testcase_index);
                        total_sweeps.fetch_add(static_cast<std::size_t>(packet.sweeps), std::memory_order_relaxed);
                        return packet;
                    });
                    output.submit(std::move(packet_future));

                    submitted_count += 1;
                    testcase_index += 1;
                }
            }

            output.close_success(submitted_count);

            PipelineReport report;
            report.testcase_count = submitted_count;
            report.emitted_matrix_count = submitted_count * 3;
            report.total_sweeps = total_sweeps.load(std::memory_order_relaxed);
            report.layout_transpose_mode = runtime_kernel_config.layout_transpose_mode;
            report.layout_transpose_min_columns = runtime_kernel_config.layout_transpose_min_columns;
            report.layout_transpose_min_elements = runtime_kernel_config.layout_transpose_min_elements;
            report.layout_transpose_auto_tuned = tuning_report.executed;
            report.layout_transpose_estimated_best_speedup = tuning_report.estimated_best_speedup;
            return report;
        }
        catch (...)
        {
            try
            {
                output.close_error(std::current_exception());
            }
            catch (...)
            {
            }
            throw;
        }
    }

    PipelineReport run_pipeline(const PipelineConfig &config)
    {
        return JacobiSvdPipeline(config).run();
    }
} // namespace jacobi::svd::pipeline
