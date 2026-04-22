#include "io.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace jacobi::svd::io
{
    namespace
    {
        /**
         * @brief *.mat 头部字节数；Header bytes of *.mat.
         */
        constexpr std::size_t kMatHeaderBytes = sizeof(MatMetaData);

        /**
         * @brief *.mat 单元素字节数；Element bytes in *.mat.
         */
        constexpr std::size_t kMatElementBytes = sizeof(std::uint64_t);

        /**
         * @brief 解码分块元素数；Chunk element count for decoding.
         */
        constexpr std::size_t kDecodeChunkElements = 32U * 1024U;

        /**
         * @brief 编码分块元素数；Chunk element count for encoding.
         */
        constexpr std::size_t kEncodeChunkElements = 32U * 1024U;

        /**
         * @brief 输出映射最小容量；Minimum mapped capacity for output.
         */
        constexpr std::size_t kMinMappedCapacity = 1U << 20U;

        /**
         * @brief 任务页锁定缓冲最小容量；Minimum pinned-task buffer capacity.
         */
        constexpr std::size_t kMinPinnedTaskCapacity = 64U * 1024U;

        /**
         * @brief 检查并计算无符号乘法；Checked multiplication for unsigned sizes.
         * @param lhs 左操作数；Left operand.
         * @param rhs 右操作数；Right operand.
         * @return 乘积；Product.
         */
        [[nodiscard]] std::size_t checked_multiply(std::size_t lhs, std::size_t rhs)
        {
            if (lhs == 0 || rhs == 0)
            {
                return 0;
            }
            if (lhs > (std::numeric_limits<std::size_t>::max() / rhs))
            {
                throw std::overflow_error("Size multiplication overflow.");
            }
            return lhs * rhs;
        }

        /**
         * @brief 检查并计算无符号加法；Checked addition for unsigned sizes.
         * @param lhs 左操作数；Left operand.
         * @param rhs 右操作数；Right operand.
         * @return 和；Sum.
         */
        [[nodiscard]] std::size_t checked_add(std::size_t lhs, std::size_t rhs)
        {
            if (lhs > (std::numeric_limits<std::size_t>::max() - rhs))
            {
                throw std::overflow_error("Size addition overflow.");
            }
            return lhs + rhs;
        }

        /**
         * @brief 64 位字节交换；Byte-swap for 64-bit integers.
         * @param value 输入值；Input value.
         * @return 字节交换后结果；Byte-swapped result.
         */
        [[nodiscard]] constexpr std::uint64_t byte_swap_u64(std::uint64_t value) noexcept
        {
#if defined(__cpp_lib_byteswap) && (__cpp_lib_byteswap >= 202110L)
            return std::byteswap(value);
#else
            return ((value & 0x00000000000000FFULL) << 56U) |
                   ((value & 0x000000000000FF00ULL) << 40U) |
                   ((value & 0x0000000000FF0000ULL) << 24U) |
                   ((value & 0x00000000FF000000ULL) << 8U) |
                   ((value & 0x000000FF00000000ULL) >> 8U) |
                   ((value & 0x0000FF0000000000ULL) >> 24U) |
                   ((value & 0x00FF000000000000ULL) >> 40U) |
                   ((value & 0xFF00000000000000ULL) >> 56U);
#endif
        }

        /**
         * @brief 主机序转网络序；Host to network byte order.
         * @param value 主机值；Host value.
         * @return 网络序值；Network-order value.
         */
        [[nodiscard]] constexpr std::uint64_t to_network_u64(std::uint64_t value) noexcept
        {
            if constexpr (std::endian::native == std::endian::little)
            {
                return byte_swap_u64(value);
            }
            return value;
        }

        /**
         * @brief 网络序转主机序；Network to host byte order.
         * @param value 网络序值；Network-order value.
         * @return 主机值；Host value.
         */
        [[nodiscard]] constexpr std::uint64_t from_network_u64(std::uint64_t value) noexcept
        {
            if constexpr (std::endian::native == std::endian::little)
            {
                return byte_swap_u64(value);
            }
            return value;
        }

        /**
         * @brief 将 double 编码为网络序；Encode double to network byte order.
         * @param value 输入浮点；Input floating-point value.
         * @return 网络序 64 位模式；Network-order 64-bit pattern.
         */
        [[nodiscard]] std::uint64_t encode_network_double(double value) noexcept
        {
            return to_network_u64(std::bit_cast<std::uint64_t>(value));
        }

        /**
         * @brief 将网络序解码为 double；Decode network byte order to double.
         * @param value 网络序 64 位模式；Network-order 64-bit pattern.
         * @return 主机浮点值；Host floating-point value.
         */
        [[nodiscard]] double decode_network_double(std::uint64_t value) noexcept
        {
            return std::bit_cast<double>(from_network_u64(value));
        }

        /**
         * @brief 验证矩阵布局；Validate matrix layout.
         * @param matrix 输入矩阵；Input matrix.
         */
        void validate_matrix_layout(const Matrix &matrix)
        {
            const std::size_t expected_count = checked_multiply(matrix.rows, matrix.columns);
            if (matrix.values.size() != expected_count)
            {
                throw std::invalid_argument("Matrix payload size does not match rows * columns.");
            }
        }

        /**
         * @brief 判断文本行是否为空白；Test whether text line is blank.
         * @param line 输入文本行；Input text line.
         * @return true 表示空白；true if blank.
         */
        [[nodiscard]] bool is_blank_line(const std::string &line)
        {
            return std::all_of(line.begin(), line.end(), [](unsigned char ch) {
                return std::isspace(ch) != 0;
            });
        }

        /**
         * @brief 解析文本行浮点值；Parse floating values from one text line.
         * @param line 输入文本行；Input text line.
         * @return 浮点序列；Floating-point sequence.
         */
        [[nodiscard]] std::vector<double> parse_txt_row(const std::string &line)
        {
            std::istringstream row_stream(line);
            row_stream.imbue(std::locale::classic());

            std::vector<double> row_values;
            double parsed = 0.0;
            while (row_stream >> parsed)
            {
                row_values.push_back(parsed);
            }

            if (row_stream.fail() && !row_stream.eof())
            {
                throw std::invalid_argument("Invalid numeric token in text matrix row.");
            }
            return row_values;
        }

        /**
         * @brief 将 size_t 安全转换为 streamsize；Safely convert size_t to streamsize.
         * @param bytes 字节数；Byte count.
         * @return streamsize 数值；Converted streamsize value.
         */
        [[nodiscard]] std::streamsize checked_to_streamsize(std::size_t bytes)
        {
            if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
            {
                throw std::overflow_error("Byte size exceeds streamsize range.");
            }
            return static_cast<std::streamsize>(bytes);
        }

        /**
         * @brief 计算增长后容量；Compute grown capacity.
         * @param current 当前容量；Current capacity.
         * @param required 需求容量；Required capacity.
         * @return 新容量；New capacity.
         */
        [[nodiscard]] std::size_t grow_capacity(std::size_t current, std::size_t required)
        {
            if (required == 0)
            {
                return 0;
            }

            std::size_t grown = (current == 0) ? kMinPinnedTaskCapacity : current;
            while (grown < required)
            {
                if (grown > (std::numeric_limits<std::size_t>::max() / 2))
                {
                    return required;
                }
                grown *= 2;
            }
            return grown;
        }

#ifdef _WIN32
        /**
         * @brief 路径转 Windows 宽字符串；Convert path to Windows wide string.
         * @param path 输入路径；Input path.
         * @return 宽字符串路径；Wide-string path.
         */
        [[nodiscard]] std::wstring to_windows_path(const std::filesystem::path &path)
        {
            return path.wstring();
        }
#endif

        /**
         * @brief 只读内存映射文件；Read-only memory-mapped file.
         */
        class MemoryMappedInputFile final
        {
        public:
            /**
             * @brief 打开并映射输入文件；Open and map input file.
             * @param path 文件路径；File path.
             */
            explicit MemoryMappedInputFile(const std::filesystem::path &path)
            {
#ifdef _WIN32
                const std::wstring wide_path = to_windows_path(path);
                file_ = ::CreateFileW(wide_path.c_str(),
                                      GENERIC_READ,
                                      FILE_SHARE_READ,
                                      nullptr,
                                      OPEN_EXISTING,
                                      FILE_ATTRIBUTE_NORMAL,
                                      nullptr);
                if (file_ == INVALID_HANDLE_VALUE)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "CreateFileW failed");
                }

                LARGE_INTEGER file_size{};
                if (::GetFileSizeEx(file_, &file_size) == 0)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "GetFileSizeEx failed");
                }
                if (file_size.QuadPart < 0)
                {
                    throw std::runtime_error("Negative file size is invalid.");
                }
                size_ = static_cast<std::size_t>(file_size.QuadPart);
                if (size_ == 0)
                {
                    return;
                }

                mapping_ = ::CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
                if (mapping_ == nullptr)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "CreateFileMappingW failed");
                }

                void *mapped = ::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
                if (mapped == nullptr)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "MapViewOfFile failed");
                }
                data_ = static_cast<const std::byte *>(mapped);
#else
                file_descriptor_ = ::open(path.c_str(), O_RDONLY);
                if (file_descriptor_ < 0)
                {
                    throw std::system_error(errno, std::system_category(), "open failed");
                }

                struct stat file_state
                {
                };
                if (::fstat(file_descriptor_, &file_state) != 0)
                {
                    throw std::system_error(errno, std::system_category(), "fstat failed");
                }
                if (file_state.st_size < 0)
                {
                    throw std::runtime_error("Negative file size is invalid.");
                }

                size_ = static_cast<std::size_t>(file_state.st_size);
                if (size_ == 0)
                {
                    return;
                }

                void *mapped = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, file_descriptor_, 0);
                if (mapped == MAP_FAILED)
                {
                    throw std::system_error(errno, std::system_category(), "mmap failed");
                }
                data_ = static_cast<const std::byte *>(mapped);
#endif
            }

            /**
             * @brief 析构释放资源；Destroy and release resources.
             */
            ~MemoryMappedInputFile()
            {
                close();
            }

            /**
             * @brief 禁止拷贝构造；Copy constructor is disabled.
             */
            MemoryMappedInputFile(const MemoryMappedInputFile &) = delete;

            /**
             * @brief 禁止拷贝赋值；Copy assignment is disabled.
             * @return 当前对象引用；Reference to current object.
             */
            MemoryMappedInputFile &operator=(const MemoryMappedInputFile &) = delete;

            /**
             * @brief 移动构造；Move constructor.
             * @param other 源对象；Source object.
             */
            MemoryMappedInputFile(MemoryMappedInputFile &&other) noexcept
            {
                move_from(std::move(other));
            }

            /**
             * @brief 移动赋值；Move assignment.
             * @param other 源对象；Source object.
             * @return 当前对象引用；Reference to current object.
             */
            MemoryMappedInputFile &operator=(MemoryMappedInputFile &&other) noexcept
            {
                if (this != &other)
                {
                    close();
                    move_from(std::move(other));
                }
                return *this;
            }

            /**
             * @brief 获取映射字节视图；Get mapped bytes.
             * @return 只读字节视图；Read-only byte span.
             */
            [[nodiscard]] std::span<const std::byte> bytes() const noexcept
            {
                return {data_, size_};
            }

        private:
            /**
             * @brief 从另一个对象移动资源；Move resources from another object.
             * @param other 源对象；Source object.
             */
            void move_from(MemoryMappedInputFile &&other) noexcept
            {
                size_ = std::exchange(other.size_, 0);
                data_ = std::exchange(other.data_, nullptr);
#ifdef _WIN32
                mapping_ = std::exchange(other.mapping_, nullptr);
                file_ = std::exchange(other.file_, INVALID_HANDLE_VALUE);
#else
                file_descriptor_ = std::exchange(other.file_descriptor_, -1);
#endif
            }

            /**
             * @brief 关闭映射并释放资源；Close mapping and release resources.
             */
            void close() noexcept
            {
#ifdef _WIN32
                if (data_ != nullptr)
                {
                    (void)::UnmapViewOfFile(data_);
                    data_ = nullptr;
                }
                if (mapping_ != nullptr)
                {
                    (void)::CloseHandle(mapping_);
                    mapping_ = nullptr;
                }
                if (file_ != INVALID_HANDLE_VALUE)
                {
                    (void)::CloseHandle(file_);
                    file_ = INVALID_HANDLE_VALUE;
                }
#else
                if (data_ != nullptr)
                {
                    (void)::munmap(const_cast<std::byte *>(data_), size_);
                    data_ = nullptr;
                }
                if (file_descriptor_ >= 0)
                {
                    (void)::close(file_descriptor_);
                    file_descriptor_ = -1;
                }
#endif
                size_ = 0;
            }

            /**
             * @brief 映射长度；Mapped length.
             */
            std::size_t size_ = 0;

            /**
             * @brief 映射地址；Mapped address.
             */
            const std::byte *data_ = nullptr;

#ifdef _WIN32
            /**
             * @brief 文件句柄；File handle.
             */
            HANDLE file_ = INVALID_HANDLE_VALUE;

            /**
             * @brief 映射句柄；Mapping handle.
             */
            HANDLE mapping_ = nullptr;
#else
            /**
             * @brief POSIX 文件描述符；POSIX file descriptor.
             */
            int file_descriptor_ = -1;
#endif
        };

        /**
         * @brief 追加式可扩展内存映射输出文件；Appendable and growable memory-mapped output file.
         */
        class AppendMappedOutputFile final
        {
        public:
            /**
             * @brief 创建输出文件；Create output file.
             * @param path 文件路径；File path.
             */
            explicit AppendMappedOutputFile(const std::filesystem::path &path)
            {
#ifdef _WIN32
                const std::wstring wide_path = to_windows_path(path);
                file_ = ::CreateFileW(wide_path.c_str(),
                                      GENERIC_READ | GENERIC_WRITE,
                                      0,
                                      nullptr,
                                      CREATE_ALWAYS,
                                      FILE_ATTRIBUTE_NORMAL,
                                      nullptr);
                if (file_ == INVALID_HANDLE_VALUE)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "CreateFileW failed");
                }
#else
                file_descriptor_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
                if (file_descriptor_ < 0)
                {
                    throw std::system_error(errno, std::system_category(), "open failed");
                }
#endif
            }

            /**
             * @brief 析构并落盘；Destroy and flush.
             */
            ~AppendMappedOutputFile()
            {
                close();
            }

            /**
             * @brief 禁止拷贝构造；Copy constructor is disabled.
             */
            AppendMappedOutputFile(const AppendMappedOutputFile &) = delete;

            /**
             * @brief 禁止拷贝赋值；Copy assignment is disabled.
             * @return 当前对象引用；Reference to current object.
             */
            AppendMappedOutputFile &operator=(const AppendMappedOutputFile &) = delete;

            /**
             * @brief 移动构造；Move constructor.
             * @param other 源对象；Source object.
             */
            AppendMappedOutputFile(AppendMappedOutputFile &&other) noexcept
            {
                move_from(std::move(other));
            }

            /**
             * @brief 移动赋值；Move assignment.
             * @param other 源对象；Source object.
             * @return 当前对象引用；Reference to current object.
             */
            AppendMappedOutputFile &operator=(AppendMappedOutputFile &&other) noexcept
            {
                if (this != &other)
                {
                    close();
                    move_from(std::move(other));
                }
                return *this;
            }

            /**
             * @brief 追加字节数据；Append byte payload.
             * @param payload 输入字节序列；Input byte sequence.
             */
            void append(std::span<const std::byte> payload)
            {
                if (payload.empty())
                {
                    return;
                }

                const std::size_t required = checked_add(size_, payload.size());
                ensure_capacity(required);
                std::memcpy(data_ + size_, payload.data(), payload.size());
                size_ = required;
            }

            /**
             * @brief 刷新已写内容；Flush written content.
             */
            void flush()
            {
                flush_mapped_prefix();
            }

        private:
            /**
             * @brief 确保映射容量；Ensure mapped capacity.
             * @param required 目标最小容量；Required minimum capacity.
             */
            void ensure_capacity(std::size_t required)
            {
                if (required <= capacity_)
                {
                    return;
                }

                std::size_t new_capacity = (capacity_ == 0) ? kMinMappedCapacity : capacity_;
                while (new_capacity < required)
                {
                    if (new_capacity > (std::numeric_limits<std::size_t>::max() / 2))
                    {
                        new_capacity = required;
                        break;
                    }
                    new_capacity *= 2;
                }
                if (new_capacity < required)
                {
                    new_capacity = required;
                }

                remap(new_capacity);
            }

            /**
             * @brief 重新映射到新容量；Remap to new capacity.
             * @param new_capacity 新容量；New capacity.
             */
            void remap(std::size_t new_capacity)
            {
                flush_mapped_prefix();
                unmap();
                resize_file(new_capacity);
                map(new_capacity);
                capacity_ = new_capacity;
            }

            /**
             * @brief 映射文件；Map file.
             * @param mapped_size 映射字节数；Mapped byte size.
             */
            void map(std::size_t mapped_size)
            {
                if (mapped_size == 0)
                {
                    data_ = nullptr;
                    return;
                }
#ifdef _WIN32
                const std::uint64_t upper = static_cast<std::uint64_t>(mapped_size) >> 32U;
                const std::uint64_t lower = static_cast<std::uint64_t>(mapped_size) & 0xFFFFFFFFULL;
                mapping_ = ::CreateFileMappingW(file_,
                                                nullptr,
                                                PAGE_READWRITE,
                                                static_cast<DWORD>(upper),
                                                static_cast<DWORD>(lower),
                                                nullptr);
                if (mapping_ == nullptr)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "CreateFileMappingW failed");
                }

                void *mapped = ::MapViewOfFile(mapping_, FILE_MAP_WRITE, 0, 0, mapped_size);
                if (mapped == nullptr)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "MapViewOfFile failed");
                }
                data_ = static_cast<std::byte *>(mapped);
#else
                void *mapped = ::mmap(nullptr, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, file_descriptor_, 0);
                if (mapped == MAP_FAILED)
                {
                    throw std::system_error(errno, std::system_category(), "mmap failed");
                }
                data_ = static_cast<std::byte *>(mapped);
#endif
            }

            /**
             * @brief 取消当前映射；Unmap current view.
             */
            void unmap() noexcept
            {
#ifdef _WIN32
                if (data_ != nullptr)
                {
                    (void)::UnmapViewOfFile(data_);
                    data_ = nullptr;
                }
                if (mapping_ != nullptr)
                {
                    (void)::CloseHandle(mapping_);
                    mapping_ = nullptr;
                }
#else
                if (data_ != nullptr)
                {
                    (void)::munmap(data_, capacity_);
                    data_ = nullptr;
                }
#endif
            }

            /**
             * @brief 调整底层文件大小；Resize underlying file.
             * @param target_size 目标文件大小；Target file size.
             */
            void resize_file(std::size_t target_size)
            {
#ifdef _WIN32
                LARGE_INTEGER position{};
                position.QuadPart = static_cast<LONGLONG>(target_size);
                if (::SetFilePointerEx(file_, position, nullptr, FILE_BEGIN) == 0)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "SetFilePointerEx failed");
                }
                if (::SetEndOfFile(file_) == 0)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "SetEndOfFile failed");
                }
#else
                if (::ftruncate(file_descriptor_, static_cast<off_t>(target_size)) != 0)
                {
                    throw std::system_error(errno, std::system_category(), "ftruncate failed");
                }
#endif
            }

            /**
             * @brief 刷新当前已写前缀；Flush currently written prefix.
             */
            void flush_mapped_prefix()
            {
                if (data_ == nullptr || size_ == 0)
                {
                    return;
                }
#ifdef _WIN32
                if (::FlushViewOfFile(data_, size_) == 0)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "FlushViewOfFile failed");
                }
                if (::FlushFileBuffers(file_) == 0)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "FlushFileBuffers failed");
                }
#else
                if (::msync(data_, size_, MS_SYNC) != 0)
                {
                    throw std::system_error(errno, std::system_category(), "msync failed");
                }
#endif
            }

            /**
             * @brief 从另一个对象移动资源；Move resources from another object.
             * @param other 源对象；Source object.
             */
            void move_from(AppendMappedOutputFile &&other) noexcept
            {
                size_ = std::exchange(other.size_, 0);
                capacity_ = std::exchange(other.capacity_, 0);
                data_ = std::exchange(other.data_, nullptr);
#ifdef _WIN32
                file_ = std::exchange(other.file_, INVALID_HANDLE_VALUE);
                mapping_ = std::exchange(other.mapping_, nullptr);
#else
                file_descriptor_ = std::exchange(other.file_descriptor_, -1);
#endif
            }

            /**
             * @brief 关闭映射和句柄；Close mapping and handles.
             */
            void close() noexcept
            {
                try
                {
                    flush_mapped_prefix();
                }
                catch (...)
                {
                }

                unmap();

#ifdef _WIN32
                if (file_ != INVALID_HANDLE_VALUE)
                {
                    LARGE_INTEGER position{};
                    position.QuadPart = static_cast<LONGLONG>(size_);
                    (void)::SetFilePointerEx(file_, position, nullptr, FILE_BEGIN);
                    (void)::SetEndOfFile(file_);
                    (void)::CloseHandle(file_);
                    file_ = INVALID_HANDLE_VALUE;
                }
#else
                if (file_descriptor_ >= 0)
                {
                    (void)::ftruncate(file_descriptor_, static_cast<off_t>(size_));
                    (void)::close(file_descriptor_);
                    file_descriptor_ = -1;
                }
#endif
                size_ = 0;
                capacity_ = 0;
            }

            /**
             * @brief 已写字节数；Written byte count.
             */
            std::size_t size_ = 0;

            /**
             * @brief 映射容量；Mapped capacity.
             */
            std::size_t capacity_ = 0;

            /**
             * @brief 映射地址；Mapped address.
             */
            std::byte *data_ = nullptr;

#ifdef _WIN32
            /**
             * @brief 文件句柄；File handle.
             */
            HANDLE file_ = INVALID_HANDLE_VALUE;

            /**
             * @brief 映射句柄；Mapping handle.
             */
            HANDLE mapping_ = nullptr;
#else
            /**
             * @brief POSIX 文件描述符；POSIX file descriptor.
             */
            int file_descriptor_ = -1;
#endif
        };
    } // namespace

    /**
     * @brief *.mat 读取器实现体；Implementation of *.mat reader.
     */
    struct MatReader::Impl final
    {
        /**
         * @brief 映射文件对象；Mapped file object.
         */
        MemoryMappedInputFile mapped_file;

        /**
         * @brief 映射字节视图；Mapped byte span.
         */
        std::span<const std::byte> bytes;

        /**
         * @brief 当前游标；Current cursor.
         */
        std::size_t cursor = 0;

        /**
         * @brief 构造实现体；Construct implementation.
         * @param path 文件路径；File path.
         */
        explicit Impl(const std::filesystem::path &path)
            : mapped_file(path), bytes(mapped_file.bytes())
        {
        }
    };

    /**
     * @brief *.mat 写入器实现体；Implementation of *.mat writer.
     */
    struct MatWriter::Impl final
    {
        /**
         * @brief 追加式映射输出文件；Appendable mapped output file.
         */
        AppendMappedOutputFile mapped_output;

        /**
         * @brief 构造实现体；Construct implementation.
         * @param path 文件路径；File path.
         */
        explicit Impl(const std::filesystem::path &path)
            : mapped_output(path)
        {
        }
    };

    /**
     * @brief *.txt 读取器实现体；Implementation of *.txt reader.
     */
    struct TxtReader::Impl final
    {
        /**
         * @brief 输入文件流；Input file stream.
         */
        std::ifstream input;

        /**
         * @brief 构造实现体；Construct implementation.
         * @param path 文件路径；File path.
         */
        explicit Impl(const std::filesystem::path &path)
            : input(path, std::ios::in | std::ios::binary)
        {
            if (!input)
            {
                throw std::runtime_error("Failed to open text matrix file for reading.");
            }
            input.imbue(std::locale::classic());
        }
    };

    /**
     * @brief *.txt 写入器实现体；Implementation of *.txt writer.
     */
    struct TxtWriter::Impl final
    {
        /**
         * @brief 输出文件流；Output file stream.
         */
        std::ofstream output;

        /**
         * @brief 是否已有写入内容；Whether any matrix has been written.
         */
        bool has_written_matrix = false;

        /**
         * @brief 构造实现体；Construct implementation.
         * @param path 文件路径；File path.
         */
        explicit Impl(const std::filesystem::path &path)
            : output(path, std::ios::out | std::ios::binary | std::ios::trunc)
        {
            if (!output)
            {
                throw std::runtime_error("Failed to open text matrix file for writing.");
            }
            output.imbue(std::locale::classic());
            output << std::setprecision(std::numeric_limits<double>::max_digits10);
        }
    };

    MatReader::MatReader(const std::filesystem::path &path)
        : impl_(std::make_unique<Impl>(path))
    {
    }

    MatReader::~MatReader() = default;

    MatReader::MatReader(MatReader &&other) noexcept = default;

    MatReader &MatReader::operator=(MatReader &&other) noexcept = default;

    MatWriter::MatWriter(const std::filesystem::path &path)
        : impl_(std::make_unique<Impl>(path))
    {
    }

    MatWriter::~MatWriter() = default;

    MatWriter::MatWriter(MatWriter &&other) noexcept = default;

    MatWriter &MatWriter::operator=(MatWriter &&other) noexcept = default;

    TxtReader::TxtReader(const std::filesystem::path &path)
        : impl_(std::make_unique<Impl>(path))
    {
    }

    TxtReader::~TxtReader() = default;

    TxtReader::TxtReader(TxtReader &&other) noexcept = default;

    TxtReader &TxtReader::operator=(TxtReader &&other) noexcept = default;

    TxtWriter::TxtWriter(const std::filesystem::path &path)
        : impl_(std::make_unique<Impl>(path))
    {
    }

    TxtWriter::~TxtWriter() = default;

    TxtWriter::TxtWriter(TxtWriter &&other) noexcept = default;

    TxtWriter &TxtWriter::operator=(TxtWriter &&other) noexcept = default;

    PinnedHostTaskBuffer::~PinnedHostTaskBuffer()
    {
        release();
    }

    PinnedHostTaskBuffer::PinnedHostTaskBuffer(PinnedHostTaskBuffer &&other) noexcept
    {
        move_from(std::move(other));
    }

    PinnedHostTaskBuffer &PinnedHostTaskBuffer::operator=(PinnedHostTaskBuffer &&other) noexcept
    {
        if (this != &other)
        {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    void PinnedHostTaskBuffer::reserve(std::size_t input_bytes, std::size_t workspace_bytes)
    {
        const std::size_t required_bytes = checked_add(input_bytes, workspace_bytes);
        if (required_bytes > capacity_bytes_)
        {
            const std::size_t new_capacity = grow_capacity(capacity_bytes_, required_bytes);
            void *new_data = nullptr;
            const cudaError_t status = ::cudaMallocHost(&new_data, new_capacity);
            if (status != cudaSuccess)
            {
                throw std::runtime_error("cudaMallocHost failed: " +
                                         std::string(::cudaGetErrorString(status)));
            }

            release();
            data_ = static_cast<std::byte *>(new_data);
            capacity_bytes_ = new_capacity;
        }

        input_size_bytes_ = input_bytes;
        workspace_size_bytes_ = workspace_bytes;
    }

    std::span<std::byte> PinnedHostTaskBuffer::mutable_input_bytes() noexcept
    {
        return {data_, input_size_bytes_};
    }

    std::span<const std::byte> PinnedHostTaskBuffer::input_bytes() const noexcept
    {
        return {data_, input_size_bytes_};
    }

    std::span<std::byte> PinnedHostTaskBuffer::mutable_workspace_bytes() noexcept
    {
        return {data_ + input_size_bytes_, workspace_size_bytes_};
    }

    std::span<const std::byte> PinnedHostTaskBuffer::workspace_bytes() const noexcept
    {
        return {data_ + input_size_bytes_, workspace_size_bytes_};
    }

    std::size_t PinnedHostTaskBuffer::capacity_bytes() const noexcept
    {
        return capacity_bytes_;
    }

    std::size_t PinnedHostTaskBuffer::input_size_bytes() const noexcept
    {
        return input_size_bytes_;
    }

    std::size_t PinnedHostTaskBuffer::workspace_size_bytes() const noexcept
    {
        return workspace_size_bytes_;
    }

    void PinnedHostTaskBuffer::release() noexcept
    {
        if (data_ != nullptr)
        {
            (void)::cudaFreeHost(data_);
            data_ = nullptr;
        }
        capacity_bytes_ = 0;
        input_size_bytes_ = 0;
        workspace_size_bytes_ = 0;
    }

    void PinnedHostTaskBuffer::move_from(PinnedHostTaskBuffer &&other) noexcept
    {
        data_ = std::exchange(other.data_, nullptr);
        capacity_bytes_ = std::exchange(other.capacity_bytes_, 0);
        input_size_bytes_ = std::exchange(other.input_size_bytes_, 0);
        workspace_size_bytes_ = std::exchange(other.workspace_size_bytes_, 0);
    }

    MatDispatchReader::MatDispatchReader(const std::filesystem::path &path)
        : input_(path, std::ios::in | std::ios::binary)
    {
        if (!input_)
        {
            throw std::runtime_error("Failed to open *.mat file for dispatch reading.");
        }
    }

    bool MatDispatchReader::read_next(MatDispatchTask &task, std::size_t workspace_bytes)
    {
        MatMetaData encoded_meta{};
        input_.read(reinterpret_cast<char *>(&encoded_meta), checked_to_streamsize(kMatHeaderBytes));

        if (input_.gcount() == 0 && input_.eof())
        {
            return false;
        }
        if (input_.gcount() != checked_to_streamsize(kMatHeaderBytes))
        {
            throw std::runtime_error("Truncated *.mat header while dispatch reading.");
        }

        const std::uint64_t rows_u64 = from_network_u64(encoded_meta.rows);
        const std::uint64_t columns_u64 = from_network_u64(encoded_meta.columns);
        if (rows_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            columns_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::overflow_error("Matrix dimensions exceed platform size_t range.");
        }

        const std::size_t rows = static_cast<std::size_t>(rows_u64);
        const std::size_t columns = static_cast<std::size_t>(columns_u64);
        const std::size_t element_count = checked_multiply(rows, columns);
        const std::size_t payload_bytes = checked_multiply(element_count, kMatElementBytes);

        task.buffer.reserve(payload_bytes, workspace_bytes);
        if (payload_bytes > 0)
        {
            std::span<std::byte> input_bytes = task.buffer.mutable_input_bytes();
            input_.read(reinterpret_cast<char *>(input_bytes.data()), checked_to_streamsize(payload_bytes));
            if (input_.gcount() != checked_to_streamsize(payload_bytes))
            {
                throw std::runtime_error("Truncated *.mat payload while dispatch reading.");
            }
        }

        task.sequence_index = next_sequence_index_;
        task.rows = rows;
        task.columns = columns;
        next_sequence_index_ = checked_add(next_sequence_index_, 1);
        return true;
    }

    Matrix decode_dispatch_task_matrix(const MatDispatchTask &task)
    {
        Matrix matrix;
        matrix.rows = task.rows;
        matrix.columns = task.columns;

        const std::size_t element_count = checked_multiply(matrix.rows, matrix.columns);
        const std::size_t payload_bytes = checked_multiply(element_count, kMatElementBytes);
        const std::span<const std::byte> payload = task.buffer.input_bytes();
        if (payload.size() != payload_bytes)
        {
            throw std::invalid_argument("Dispatch payload size does not match rows * columns.");
        }

        matrix.values.resize(element_count);
        for (std::size_t index = 0; index < element_count; ++index)
        {
            std::uint64_t encoded = 0;
            std::memcpy(&encoded, payload.data() + checked_multiply(index, kMatElementBytes), kMatElementBytes);
            matrix.values[index] = decode_network_double(encoded);
        }
        return matrix;
    }

    MatFilePolicy::Reader MatFilePolicy::open_reader(const std::filesystem::path &path)
    {
        return MatReader(path);
    }

    MatFilePolicy::Writer MatFilePolicy::open_writer(const std::filesystem::path &path)
    {
        return MatWriter(path);
    }

    bool MatFilePolicy::read_next(Reader &reader, Matrix &matrix)
    {
        if (reader.impl_ == nullptr)
        {
            throw std::runtime_error("MatReader is not initialized.");
        }

        auto &state = *reader.impl_;
        if (state.cursor == state.bytes.size())
        {
            return false;
        }

        const std::size_t remaining = state.bytes.size() - state.cursor;
        if (remaining < kMatHeaderBytes)
        {
            throw std::runtime_error("Truncated *.mat header.");
        }

        MatMetaData encoded_meta{};
        std::memcpy(&encoded_meta, state.bytes.data() + state.cursor, kMatHeaderBytes);
        state.cursor = checked_add(state.cursor, kMatHeaderBytes);

        const std::uint64_t rows_u64 = from_network_u64(encoded_meta.rows);
        const std::uint64_t columns_u64 = from_network_u64(encoded_meta.columns);

        if (rows_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            columns_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::overflow_error("Matrix dimensions exceed platform size_t range.");
        }

        matrix.rows = static_cast<std::size_t>(rows_u64);
        matrix.columns = static_cast<std::size_t>(columns_u64);

        const std::size_t element_count = checked_multiply(matrix.rows, matrix.columns);
        const std::size_t payload_bytes = checked_multiply(element_count, kMatElementBytes);

        if ((state.bytes.size() - state.cursor) < payload_bytes)
        {
            throw std::runtime_error("Truncated *.mat payload.");
        }

        matrix.values.resize(element_count);

        std::size_t processed = 0;
        while (processed < element_count)
        {
            const std::size_t chunk_count = std::min(kDecodeChunkElements, element_count - processed);
            const std::byte *chunk_base = state.bytes.data() + state.cursor + checked_multiply(processed, kMatElementBytes);

            for (std::size_t index = 0; index < chunk_count; ++index)
            {
                std::uint64_t encoded = 0;
                std::memcpy(&encoded, chunk_base + checked_multiply(index, kMatElementBytes), kMatElementBytes);
                matrix.values[processed + index] = decode_network_double(encoded);
            }

            processed = checked_add(processed, chunk_count);
        }

        state.cursor = checked_add(state.cursor, payload_bytes);
        return true;
    }

    void MatFilePolicy::write_next(Writer &writer, const Matrix &matrix)
    {
        if (writer.impl_ == nullptr)
        {
            throw std::runtime_error("MatWriter is not initialized.");
        }

        validate_matrix_layout(matrix);
        if (matrix.rows > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()) ||
            matrix.columns > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()))
        {
            throw std::overflow_error("Matrix dimensions exceed *.mat uint64 metadata capacity.");
        }

        MatMetaData encoded_meta{
            .rows = to_network_u64(static_cast<std::uint64_t>(matrix.rows)),
            .columns = to_network_u64(static_cast<std::uint64_t>(matrix.columns)),
        };

        const std::span<const MatMetaData> header_span(&encoded_meta, 1);
        const std::span<const std::byte> header_bytes = std::as_bytes(header_span);
        writer.impl_->mapped_output.append(header_bytes);

        std::vector<std::uint64_t> encoded_chunk;
        encoded_chunk.reserve(kEncodeChunkElements);

        std::size_t processed = 0;
        while (processed < matrix.values.size())
        {
            const std::size_t chunk_count = std::min(kEncodeChunkElements, matrix.values.size() - processed);
            encoded_chunk.resize(chunk_count);
            for (std::size_t index = 0; index < chunk_count; ++index)
            {
                encoded_chunk[index] = encode_network_double(matrix.values[processed + index]);
            }

            const std::span<const std::uint64_t> chunk_span(encoded_chunk.data(), encoded_chunk.size());
            const std::span<const std::byte> chunk_bytes = std::as_bytes(chunk_span);
            writer.impl_->mapped_output.append(chunk_bytes);
            processed = checked_add(processed, chunk_count);
        }
    }

    void MatFilePolicy::flush(Writer &writer)
    {
        if (writer.impl_ == nullptr)
        {
            throw std::runtime_error("MatWriter is not initialized.");
        }
        writer.impl_->mapped_output.flush();
    }

    std::vector<Matrix> MatFilePolicy::read(const std::filesystem::path &path)
    {
        MatInputStream stream(path);
        return stream.read_all();
    }

    void MatFilePolicy::write(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        MatOutputStream stream(path);
        stream.write_all(matrices);
    }

    TxtFilePolicy::Reader TxtFilePolicy::open_reader(const std::filesystem::path &path)
    {
        return TxtReader(path);
    }

    TxtFilePolicy::Writer TxtFilePolicy::open_writer(const std::filesystem::path &path)
    {
        return TxtWriter(path);
    }

    bool TxtFilePolicy::read_next(Reader &reader, Matrix &matrix)
    {
        if (reader.impl_ == nullptr)
        {
            throw std::runtime_error("TxtReader is not initialized.");
        }

        auto &input = reader.impl_->input;
        std::vector<double> values;
        std::size_t rows = 0;
        std::size_t columns = 0;
        bool in_matrix = false;

        std::string line;
        while (std::getline(input, line))
        {
            if (is_blank_line(line))
            {
                if (!in_matrix)
                {
                    continue;
                }

                matrix.rows = rows;
                matrix.columns = columns;
                matrix.values = std::move(values);
                return true;
            }

            std::vector<double> row_values = parse_txt_row(line);
            if (row_values.empty())
            {
                throw std::invalid_argument("Text matrix row cannot be empty when line is non-blank.");
            }

            if (!in_matrix)
            {
                in_matrix = true;
                columns = row_values.size();
            }
            else if (row_values.size() != columns)
            {
                throw std::invalid_argument("Inconsistent column count in text matrix block.");
            }

            values.insert(values.end(), row_values.begin(), row_values.end());
            rows = checked_add(rows, 1);
        }

        if (!input.eof())
        {
            throw std::runtime_error("Failed while reading text matrix file.");
        }

        if (!in_matrix)
        {
            return false;
        }

        matrix.rows = rows;
        matrix.columns = columns;
        matrix.values = std::move(values);
        return true;
    }

    void TxtFilePolicy::write_next(Writer &writer, const Matrix &matrix)
    {
        if (writer.impl_ == nullptr)
        {
            throw std::runtime_error("TxtWriter is not initialized.");
        }

        validate_matrix_layout(matrix);
        auto &state = *writer.impl_;

        if (state.has_written_matrix)
        {
            state.output << '\n';
        }

        for (std::size_t row = 0; row < matrix.rows; ++row)
        {
            for (std::size_t column = 0; column < matrix.columns; ++column)
            {
                if (column > 0)
                {
                    state.output << ' ';
                }
                const std::size_t index = checked_add(checked_multiply(row, matrix.columns), column);
                state.output << matrix.values[index];
            }
            state.output << '\n';
        }

        if (!state.output)
        {
            throw std::runtime_error("Failed while writing text matrix file.");
        }

        state.has_written_matrix = true;
    }

    void TxtFilePolicy::flush(Writer &writer)
    {
        if (writer.impl_ == nullptr)
        {
            throw std::runtime_error("TxtWriter is not initialized.");
        }

        writer.impl_->output.flush();
        if (!writer.impl_->output)
        {
            throw std::runtime_error("Failed to flush text matrix file.");
        }
    }

    std::vector<Matrix> TxtFilePolicy::read(const std::filesystem::path &path)
    {
        TxtInputStream stream(path);
        return stream.read_all();
    }

    void TxtFilePolicy::write(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        TxtOutputStream stream(path);
        stream.write_all(matrices);
    }

    std::vector<Matrix> read_mat_file(const std::filesystem::path &path)
    {
        return MatFilePolicy::read(path);
    }

    void write_mat_file(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        MatFilePolicy::write(path, matrices);
    }

    std::vector<Matrix> read_txt_file(const std::filesystem::path &path)
    {
        return TxtFilePolicy::read(path);
    }

    void write_txt_file(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        TxtFilePolicy::write(path, matrices);
    }
} // namespace jacobi::svd::io
