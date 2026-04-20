#include "io.hpp"

#include <algorithm>
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

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#if defined(_MSC_VER)
#include <intrin.h>
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
         * @brief *.mat 元数据字节数常量；Byte size constant of *.mat metadata.
         */
        constexpr std::size_t kMatHeaderBytes = sizeof(MatMetaData);

        /**
         * @brief double 存储字节数常量；Byte size constant for double payload.
         */
        constexpr std::size_t kMatElementBytes = sizeof(std::uint64_t);

        /**
         * @brief 检查并计算无符号乘法；Checked multiplication for unsigned sizes.
         * @param lhs 左操作数；Left operand.
         * @param rhs 右操作数；Right operand.
         * @return 乘积；Product.
         * @throw std::overflow_error 发生溢出时抛出；Throws when overflow happens.
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
         * @throw std::overflow_error 发生溢出时抛出；Throws when overflow happens.
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
         * @return 字节交换后的结果；Byte-swapped result.
         */
        [[nodiscard]] constexpr std::uint64_t byte_swap_u64(std::uint64_t value) noexcept
        {
#if defined(__cpp_lib_byteswap) && (__cpp_lib_byteswap >= 202110L)
            return std::byteswap(value);
#elif defined(_MSC_VER)
            return static_cast<std::uint64_t>(_byteswap_uint64(value));
#elif defined(__GNUC__) || defined(__clang__)
            return static_cast<std::uint64_t>(__builtin_bswap64(value));
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
         * @brief 将主机端 64 位整数转换为网络字节序；Convert host uint64 to network byte order.
         * @param value 主机端整数；Host-side integer.
         * @return 网络序整数；Network-order integer.
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
         * @brief 将网络序 64 位整数转换为主机字节序；Convert network-order uint64 to host byte order.
         * @param value 网络序整数；Network-order integer.
         * @return 主机端整数；Host-side integer.
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
         * @brief 将 double 编码为网络字节序比特；Encode double bits as network byte order.
         * @param value 主机端浮点数；Host-side floating value.
         * @return 网络序比特模式；Network-order bit pattern.
         */
        [[nodiscard]] std::uint64_t encode_network_double(double value) noexcept
        {
            const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
            return to_network_u64(bits);
        }

        /**
         * @brief 将网络字节序比特解码为 double；Decode network-order bits into double.
         * @param value 网络序比特模式；Network-order bit pattern.
         * @return 主机端浮点数；Host-side floating value.
         */
        [[nodiscard]] double decode_network_double(std::uint64_t value) noexcept
        {
            const std::uint64_t bits = from_network_u64(value);
            return std::bit_cast<double>(bits);
        }

        /**
         * @brief 将文件路径转换为 Windows 宽字符路径；Convert path to Windows wide path.
         * @param path 输入路径；Input path.
         * @return Windows 宽字符串路径；Windows wide string path.
         */
        [[nodiscard]] std::wstring to_windows_path(const std::filesystem::path &path)
        {
#ifdef _WIN32
            return path.wstring();
#else
            return {};
#endif
        }

        /**
         * @brief 内存映射输入文件；Memory-mapped input file.
         */
        class MemoryMappedInputFile final
        {
        public:
            /**
             * @brief 打开并映射只读文件；Open and map a read-only file.
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

                void *mapped_view = ::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
                if (mapped_view == nullptr)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "MapViewOfFile failed");
                }
                data_ = static_cast<const std::byte *>(mapped_view);
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

                void *mapped_view = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, file_descriptor_, 0);
                if (mapped_view == MAP_FAILED)
                {
                    throw std::system_error(errno, std::system_category(), "mmap failed");
                }
                data_ = static_cast<const std::byte *>(mapped_view);
#endif
            }

            /**
             * @brief 析构并释放映射资源；Destroy and release mapped resources.
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
             * @brief 获取只读字节视图；Get read-only byte span.
             * @return 只读字节序列；Read-only byte sequence.
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
             * @brief 关闭并回收资源；Close and reclaim resources.
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
             * @brief 映射字节数；Mapped byte count.
             */
            std::size_t size_ = 0;

            /**
             * @brief 映射内存首地址；Mapped memory address.
             */
            const std::byte *data_ = nullptr;

#ifdef _WIN32
            /**
             * @brief 文件句柄；File handle.
             */
            HANDLE file_ = INVALID_HANDLE_VALUE;

            /**
             * @brief 文件映射句柄；File mapping handle.
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
         * @brief 内存映射输出文件；Memory-mapped output file.
         */
        class MemoryMappedOutputFile final
        {
        public:
            /**
             * @brief 创建并映射可写文件；Create and map a writable file.
             * @param path 文件路径；File path.
             * @param size 输出文件大小；Output file size.
             */
            MemoryMappedOutputFile(const std::filesystem::path &path, std::size_t size)
                : size_(size)
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

                LARGE_INTEGER length{};
                length.QuadPart = static_cast<LONGLONG>(size_);
                if (::SetFilePointerEx(file_, length, nullptr, FILE_BEGIN) == 0)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "SetFilePointerEx failed");
                }
                if (::SetEndOfFile(file_) == 0)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "SetEndOfFile failed");
                }

                if (size_ == 0)
                {
                    return;
                }

                const std::uint64_t upper = static_cast<std::uint64_t>(size_) >> 32U;
                const std::uint64_t lower = static_cast<std::uint64_t>(size_) & 0xFFFFFFFFULL;
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

                void *mapped_view = ::MapViewOfFile(mapping_, FILE_MAP_WRITE, 0, 0, size_);
                if (mapped_view == nullptr)
                {
                    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "MapViewOfFile failed");
                }
                data_ = static_cast<std::byte *>(mapped_view);
#else
                file_descriptor_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
                if (file_descriptor_ < 0)
                {
                    throw std::system_error(errno, std::system_category(), "open failed");
                }

                if (::ftruncate(file_descriptor_, static_cast<off_t>(size_)) != 0)
                {
                    throw std::system_error(errno, std::system_category(), "ftruncate failed");
                }

                if (size_ == 0)
                {
                    return;
                }

                void *mapped_view = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, file_descriptor_, 0);
                if (mapped_view == MAP_FAILED)
                {
                    throw std::system_error(errno, std::system_category(), "mmap failed");
                }
                data_ = static_cast<std::byte *>(mapped_view);
#endif
            }

            /**
             * @brief 析构并释放资源；Destroy and release resources.
             */
            ~MemoryMappedOutputFile()
            {
                close();
            }

            /**
             * @brief 禁止拷贝构造；Copy constructor is disabled.
             */
            MemoryMappedOutputFile(const MemoryMappedOutputFile &) = delete;

            /**
             * @brief 禁止拷贝赋值；Copy assignment is disabled.
             * @return 当前对象引用；Reference to current object.
             */
            MemoryMappedOutputFile &operator=(const MemoryMappedOutputFile &) = delete;

            /**
             * @brief 移动构造；Move constructor.
             * @param other 源对象；Source object.
             */
            MemoryMappedOutputFile(MemoryMappedOutputFile &&other) noexcept
            {
                move_from(std::move(other));
            }

            /**
             * @brief 移动赋值；Move assignment.
             * @param other 源对象；Source object.
             * @return 当前对象引用；Reference to current object.
             */
            MemoryMappedOutputFile &operator=(MemoryMappedOutputFile &&other) noexcept
            {
                if (this != &other)
                {
                    close();
                    move_from(std::move(other));
                }
                return *this;
            }

            /**
             * @brief 获取可写字节视图；Get writable byte span.
             * @return 可写字节序列；Writable byte sequence.
             */
            [[nodiscard]] std::span<std::byte> bytes() noexcept
            {
                return {data_, size_};
            }

            /**
             * @brief 刷新映射并落盘；Flush mapped bytes to storage.
             */
            void flush()
            {
                if (size_ == 0)
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

        private:
            /**
             * @brief 从另一个对象移动资源；Move resources from another object.
             * @param other 源对象；Source object.
             */
            void move_from(MemoryMappedOutputFile &&other) noexcept
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
             * @brief 关闭并回收资源；Close and reclaim resources.
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
                    (void)::munmap(data_, size_);
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
             * @brief 映射字节数；Mapped byte count.
             */
            std::size_t size_ = 0;

            /**
             * @brief 映射内存首地址；Mapped memory address.
             */
            std::byte *data_ = nullptr;

#ifdef _WIN32
            /**
             * @brief 文件句柄；File handle.
             */
            HANDLE file_ = INVALID_HANDLE_VALUE;

            /**
             * @brief 文件映射句柄；File mapping handle.
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
         * @brief 验证矩阵维度与数据长度；Validate matrix shape and payload size.
         * @param matrix 待验证矩阵；Matrix to validate.
         * @throw std::invalid_argument 数据与形状不一致时抛出；Throws when payload mismatches shape.
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
         * @brief 判断文本行是否为空白；Test whether a text line is blank.
         * @param line 文本行；Text line.
         * @return 是否为空白；Whether line is blank.
         */
        [[nodiscard]] bool is_blank_line(const std::string &line)
        {
            return std::all_of(line.begin(), line.end(), [](unsigned char ch) {
                return std::isspace(ch) != 0;
            });
        }

        /**
         * @brief 解析单行浮点数；Parse one row of floating-point values.
         * @param line 输入文本行；Input line.
         * @return 行向量；Row vector.
         * @throw std::invalid_argument 行中存在非法 token 时抛出；Throws when invalid tokens exist.
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
    } // namespace

    std::vector<Matrix> MatFilePolicy::read(const std::filesystem::path &path)
    {
        return read_mat_file(path);
    }

    void MatFilePolicy::write(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        write_mat_file(path, matrices);
    }

    std::vector<Matrix> TxtFilePolicy::read(const std::filesystem::path &path)
    {
        return read_txt_file(path);
    }

    void TxtFilePolicy::write(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        write_txt_file(path, matrices);
    }

    std::vector<Matrix> read_mat_file(const std::filesystem::path &path)
    {
        MemoryMappedInputFile input(path);
        const std::span<const std::byte> bytes = input.bytes();

        std::vector<Matrix> matrices;
        std::size_t cursor = 0;

        while (cursor < bytes.size())
        {
            const std::size_t remaining = bytes.size() - cursor;
            if (remaining < kMatHeaderBytes)
            {
                throw std::runtime_error("Truncated *.mat header.");
            }

            MatMetaData encoded_meta{};
            std::memcpy(&encoded_meta, bytes.data() + cursor, kMatHeaderBytes);
            cursor = checked_add(cursor, kMatHeaderBytes);

            const std::uint64_t rows_u64 = from_network_u64(encoded_meta.rows);
            const std::uint64_t columns_u64 = from_network_u64(encoded_meta.columns);

            if (rows_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
                columns_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
            {
                throw std::overflow_error("Matrix dimensions exceed platform size_t range.");
            }

            Matrix matrix;
            matrix.rows = static_cast<std::size_t>(rows_u64);
            matrix.columns = static_cast<std::size_t>(columns_u64);

            const std::size_t element_count = checked_multiply(matrix.rows, matrix.columns);
            const std::size_t payload_bytes = checked_multiply(element_count, kMatElementBytes);

            if ((bytes.size() - cursor) < payload_bytes)
            {
                throw std::runtime_error("Truncated *.mat payload.");
            }

            matrix.values.resize(element_count);
            for (std::size_t index = 0; index < element_count; ++index)
            {
                std::uint64_t encoded_value = 0;
                std::memcpy(&encoded_value,
                            bytes.data() + cursor + checked_multiply(index, kMatElementBytes),
                            kMatElementBytes);
                matrix.values[index] = decode_network_double(encoded_value);
            }

            cursor = checked_add(cursor, payload_bytes);
            matrices.push_back(std::move(matrix));
        }

        return matrices;
    }

    void write_mat_file(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        std::size_t total_bytes = 0;
        for (const Matrix &matrix : matrices)
        {
            validate_matrix_layout(matrix);
            if (matrix.rows > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()) ||
                matrix.columns > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()))
            {
                throw std::overflow_error("Matrix dimensions exceed *.mat uint64 metadata capacity.");
            }

            const std::size_t payload_bytes = checked_multiply(matrix.values.size(), kMatElementBytes);
            total_bytes = checked_add(total_bytes, checked_add(kMatHeaderBytes, payload_bytes));
        }

        MemoryMappedOutputFile output(path, total_bytes);
        std::span<std::byte> bytes = output.bytes();
        std::size_t cursor = 0;

        for (const Matrix &matrix : matrices)
        {
            MatMetaData encoded_meta{
                .rows = to_network_u64(static_cast<std::uint64_t>(matrix.rows)),
                .columns = to_network_u64(static_cast<std::uint64_t>(matrix.columns)),
            };
            std::memcpy(bytes.data() + cursor, &encoded_meta, kMatHeaderBytes);
            cursor = checked_add(cursor, kMatHeaderBytes);

            for (double value : matrix.values)
            {
                const std::uint64_t encoded_value = encode_network_double(value);
                std::memcpy(bytes.data() + cursor, &encoded_value, kMatElementBytes);
                cursor = checked_add(cursor, kMatElementBytes);
            }
        }

        output.flush();
    }

    std::vector<Matrix> read_txt_file(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::in | std::ios::binary);
        if (!input)
        {
            throw std::runtime_error("Failed to open text matrix file for reading.");
        }
        input.imbue(std::locale::classic());

        std::vector<Matrix> matrices;
        std::vector<std::vector<double>> row_buffer;
        std::size_t expected_columns = 0;

        const auto flush_one_matrix = [&]() {
            if (row_buffer.empty())
            {
                return;
            }

            Matrix matrix;
            matrix.rows = row_buffer.size();
            matrix.columns = expected_columns;
            matrix.values.reserve(checked_multiply(matrix.rows, matrix.columns));

            for (const std::vector<double> &row : row_buffer)
            {
                matrix.values.insert(matrix.values.end(), row.begin(), row.end());
            }

            matrices.push_back(std::move(matrix));
            row_buffer.clear();
            expected_columns = 0;
        };

        std::string line;
        while (std::getline(input, line))
        {
            if (is_blank_line(line))
            {
                flush_one_matrix();
                continue;
            }

            std::vector<double> row = parse_txt_row(line);
            if (row.empty())
            {
                throw std::invalid_argument("Text matrix row cannot be empty when line is non-blank.");
            }

            if (expected_columns == 0)
            {
                expected_columns = row.size();
            }
            else if (row.size() != expected_columns)
            {
                throw std::invalid_argument("Inconsistent column count in text matrix block.");
            }

            row_buffer.push_back(std::move(row));
        }

        if (!input.eof())
        {
            throw std::runtime_error("Failed while reading text matrix file.");
        }

        flush_one_matrix();
        return matrices;
    }

    void write_txt_file(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        std::ofstream output(path, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error("Failed to open text matrix file for writing.");
        }
        output.imbue(std::locale::classic());
        output << std::setprecision(std::numeric_limits<double>::max_digits10);

        for (std::size_t matrix_index = 0; matrix_index < matrices.size(); ++matrix_index)
        {
            const Matrix &matrix = matrices[matrix_index];
            validate_matrix_layout(matrix);

            for (std::size_t row = 0; row < matrix.rows; ++row)
            {
                for (std::size_t column = 0; column < matrix.columns; ++column)
                {
                    if (column > 0)
                    {
                        output << ' ';
                    }
                    const std::size_t index = checked_add(checked_multiply(row, matrix.columns), column);
                    output << matrix.values[index];
                }
                output << '\n';
            }

            if ((matrix_index + 1) < matrices.size())
            {
                output << '\n';
            }
        }

        if (!output)
        {
            throw std::runtime_error("Failed while writing text matrix file.");
        }
    }

} // namespace jacobi::svd::io
