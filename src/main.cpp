#include "pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace jacobi::svd::cli
{
    /**
     * @brief CLI 参数错误异常；CLI argument error exception.
     */
    class CliArgumentError final : public std::invalid_argument
    {
    public:
        /**
         * @brief 构造参数错误异常；Construct argument error exception.
         * @param message 错误消息；Error message.
         */
        explicit CliArgumentError(const std::string &message)
            : std::invalid_argument(message)
        {
        }
    };

    /**
     * @brief CLI 解析动作；CLI parse action.
     */
    enum class ParseAction
    {
        /**
         * @brief 正常执行 pipeline；Execute pipeline.
         */
        run,

        /**
         * @brief 输出帮助信息；Print help text.
         */
        help,

        /**
         * @brief 输出版本信息；Print version.
         */
        version
    };

    /**
     * @brief CLI 运行选项；CLI runtime options.
     */
    struct CliOptions final
    {
        /**
         * @brief 输入文件路径；Input file path.
         */
        std::filesystem::path input_path;

        /**
         * @brief 输出文件路径；Output file path.
         */
        std::filesystem::path output_path;

        /**
         * @brief 输入格式；Input format.
         */
        pipeline::MatrixFileFormat input_format = pipeline::MatrixFileFormat::auto_detect;

        /**
         * @brief 输出格式；Output format.
         */
        pipeline::MatrixFileFormat output_format = pipeline::MatrixFileFormat::auto_detect;

        /**
         * @brief 收敛阈值 epsilon；Convergence tolerance epsilon.
         */
        double epsilon = 1.0e-9;

        /**
         * @brief 最大 sweep 次数；Maximum sweep count.
         */
        int max_sweeps = 128;

        /**
         * @brief 每个 block 的线程数；Threads per CUDA block.
         */
        int threads_per_block = 256;

        /**
         * @brief 布局转置策略；Layout-transpose policy.
         */
        LayoutTransposeMode layout_transpose_mode = LayoutTransposeMode::auto_select;

        /**
         * @brief 自动策略下布局转置最小列数阈值；Auto-mode minimum-column threshold for layout transpose.
         */
        int layout_transpose_min_columns = 16;

        /**
         * @brief 自动策略下布局转置最小元素阈值；Auto-mode minimum-element threshold for layout transpose.
         */
        std::size_t layout_transpose_min_elements = 4096;

        /**
         * @brief 是否执行布局转置阈值自动调优；Whether to run layout-transpose threshold auto-tuning.
         */
        bool layout_transpose_auto_tune = false;

        /**
         * @brief 自动调优时每个尺寸的重复次数；Per-size repetitions during auto-tuning.
         */
        int layout_transpose_benchmark_repetitions = 2;

        /**
         * @brief 自动调优时每次基准的 sweep 上限；Sweep cap per benchmark run during auto-tuning.
         */
        int layout_transpose_benchmark_sweeps = 8;

        /**
         * @brief 输出队列容量；Output queue capacity.
         */
        std::size_t queue_capacity = 4;

        /**
         * @brief 鏄惁寮哄埗瑕嗙洊宸叉湁杈撳嚭锛沇hether to force overwriting existing output.
         */
        bool force_overwrite = false;

        /**
         * @brief 仅展示配置，不执行；Print configuration only without execution.
         */
        bool dry_run = false;

        /**
         * @brief 鏄惁杈撳嚭鐢熸晥閰嶇疆锛沇hether to print effective configuration before execution.
         */
        bool print_config = false;

        /**
         * @brief 是否输出 JSON 报告；Whether to emit JSON report.
         */
        bool json_report = false;

        /**
         * @brief 是否静默文本摘要；Whether to suppress text summary.
         */
        bool quiet = false;
    };

    /**
     * @brief CLI 解析结果；CLI parse result.
     */
    struct ParseResult final
    {
        /**
         * @brief 解析动作；Requested action.
         */
        ParseAction action = ParseAction::run;

        /**
         * @brief 解析得到的选项；Parsed options.
         */
        CliOptions options{};
    };

    /**
     * @brief 命令行选项标识；Identifier of one command-line option.
     */
    enum class OptionId
    {
        /**
         * @brief 输入路径选项；Input-path option.
         */
        input,

        /**
         * @brief 输出路径选项；Output-path option.
         */
        output,

        /**
         * @brief 输入格式选项；Input-format option.
         */
        input_format,

        /**
         * @brief 输出格式选项；Output-format option.
         */
        output_format,

        /**
         * @brief 统一格式选项（同时作用输入/输出）；Unified format option for both input/output.
         */
        format,

        /**
         * @brief epsilon 选项；epsilon option.
         */
        epsilon,

        /**
         * @brief 最大 sweep 选项；Maximum sweep option.
         */
        max_sweeps,

        /**
         * @brief 线程数选项；Threads-per-block option.
         */
        threads_per_block,

        /**
         * @brief 队列容量选项；Queue-capacity option.
         */
        queue_capacity,

        /**
         * @brief 布局转置策略选项；Layout-transpose policy option.
         */
        layout_transpose_mode,

        /**
         * @brief 布局转置最小列数阈值选项；Layout-transpose minimum-column threshold option.
         */
        layout_transpose_min_columns,

        /**
         * @brief 布局转置最小元素阈值选项；Layout-transpose minimum-element threshold option.
         */
        layout_transpose_min_elements,

        /**
         * @brief 布局转置阈值自动调优选项；Layout-transpose threshold auto-tuning option.
         */
        layout_transpose_auto_tune,

        /**
         * @brief 自动调优重复次数选项；Auto-tuning repetitions option.
         */
        layout_transpose_benchmark_repetitions,

        /**
         * @brief 自动调优 sweep 上限选项；Auto-tuning sweep-cap option.
         */
        layout_transpose_benchmark_sweeps,

        /**
         * @brief force 閫夐」锛沠orce-overwrite option.
         */
        force,

        /**
         * @brief dry-run 选项；dry-run option.
         */
        dry_run,

        /**
         * @brief print-config 閫夐」锛沜rint-config option.
         */
        print_config,

        /**
         * @brief JSON 报告选项；JSON-report option.
         */
        json_report,

        /**
         * @brief 静默选项；Quiet option.
         */
        quiet,

        /**
         * @brief 帮助选项；Help option.
         */
        help,

        /**
         * @brief 版本选项；Version option.
         */
        version
    };

    /**
     * @brief 单个选项定义；Single option definition.
     */
    struct OptionDefinition final
    {
        /**
         * @brief 选项标识；Option identifier.
         */
        OptionId id = OptionId::help;

        /**
         * @brief 长选项名（不含 `--`）；Long name without `--`.
         */
        std::string_view long_name{};

        /**
         * @brief 短选项名（不含 `-`，`\0` 表示无）；Short name without `-`, `\0` if absent.
         */
        char short_name = '\0';

        /**
         * @brief 是否需要参数值；Whether this option requires a value.
         */
        bool requires_value = false;

        /**
         * @brief 参数占位文本；Value placeholder text.
         */
        std::string_view value_hint{};

        /**
         * @brief 帮助说明；Help description.
         */
        std::string_view description{};
    };

    /**
     * @brief CLI 参数解析器；CLI argument parser.
     */
    class ArgParser final
    {
    public:
        /**
         * @brief 构造解析器并注册选项；Construct parser and register options.
         */
        ArgParser();

        /**
         * @brief 解析命令行参数；Parse command-line arguments.
         * @param argc 参数个数；Argument count.
         * @param argv 参数数组；Argument vector.
         * @return 解析结果；Parse result.
         */
        [[nodiscard]] ParseResult parse(int argc, char *const argv[]) const;

        /**
         * @brief 生成帮助文本；Build help message.
         * @param executable 可执行文件名；Executable name.
         * @return 帮助文本；Help text.
         */
        [[nodiscard]] std::string help_message(std::string_view executable) const;

    private:
        /**
         * @brief 解析长选项；Parse one long option token.
         * @param token 当前 token；Current token.
         * @param argv 参数数组；Argument vector.
         * @param argc 参数个数；Argument count.
         * @param index 当前索引（可前移）；Current index (can advance).
         * @param result 解析结果；Parse result.
         */
        void parse_long_option(std::string_view token,
                               char *const argv[],
                               int argc,
                               std::size_t &index,
                               ParseResult &result) const;

        /**
         * @brief 解析短选项；Parse one short-option token.
         * @param token 当前 token；Current token.
         * @param argv 参数数组；Argument vector.
         * @param argc 参数个数；Argument count.
         * @param index 当前索引（可前移）；Current index (can advance).
         * @param result 解析结果；Parse result.
         */
        void parse_short_options(std::string_view token,
                                 char *const argv[],
                                 int argc,
                                 std::size_t &index,
                                 ParseResult &result) const;

        /**
         * @brief 应用选项到结果对象；Apply one option to parse result.
         * @param option 选项定义；Option definition.
         * @param value 选项值（若有）；Option value if any.
         * @param result 解析结果；Parse result.
         */
        void apply_option(const OptionDefinition &option,
                          const std::optional<std::string_view> &value,
                          ParseResult &result) const;

        /**
         * @brief 查找长选项定义；Find option by long name.
         * @param name 长选项名；Long option name.
         * @return 匹配定义指针；Pointer to matching definition.
         */
        [[nodiscard]] const OptionDefinition *find_long_option(std::string_view name) const;

        /**
         * @brief 查找短选项定义；Find option by short name.
         * @param name 短选项字符；Short option character.
         * @return 匹配定义指针；Pointer to matching definition.
         */
        [[nodiscard]] const OptionDefinition *find_short_option(char name) const;

        /**
         * @brief 从后续参数中取值；Consume next argv token as value.
         * @param option_name 选项名（用于报错）；Option name for diagnostics.
         * @param argv 参数数组；Argument vector.
         * @param argc 参数个数；Argument count.
         * @param index 当前索引（可前移）；Current index (can advance).
         * @return 选项值；Consumed option value.
         */
        [[nodiscard]] std::string_view consume_next_value(std::string_view option_name,
                                                          char *const argv[],
                                                          int argc,
                                                          std::size_t &index) const;

        /**
         * @brief 选项定义列表；Registered option definitions.
         */
        std::vector<OptionDefinition> definitions_;
    };

    /**
     * @brief 统一小写化（ASCII）；Convert text to lowercase (ASCII).
     * @param text 输入文本；Input text.
     * @return 小写文本；Lowercased text.
     */
    [[nodiscard]] std::string to_lower_ascii(std::string_view text)
    {
        std::string lowered(text);
        std::transform(lowered.begin(),
                       lowered.end(),
                       lowered.begin(),
                       [](const unsigned char ch) {
                           return static_cast<char>(std::tolower(ch));
                       });
        return lowered;
    }

    /**
     * @brief 解析矩阵格式文本；Parse matrix format string.
     * @param raw 文本值；Raw text value.
     * @param option_name 选项名；Option name.
     * @return 文件格式枚举；Matrix file format enum.
     */
    [[nodiscard]] pipeline::MatrixFileFormat parse_matrix_format(std::string_view raw, std::string_view option_name)
    {
        const std::string lowered = to_lower_ascii(raw);
        if (lowered == "auto")
        {
            return pipeline::MatrixFileFormat::auto_detect;
        }
        if (lowered == "mat")
        {
            return pipeline::MatrixFileFormat::mat;
        }
        if (lowered == "txt")
        {
            return pipeline::MatrixFileFormat::txt;
        }

        throw CliArgumentError("Invalid value for --" + std::string(option_name) +
                               ": expected one of {auto, mat, txt}.");
    }

    /**
     * @brief 解析布局转置策略文本；Parse layout-transpose mode string.
     * @param raw 文本值；Raw text value.
     * @param option_name 选项名；Option name.
     * @return 布局转置策略；Layout-transpose mode.
     */
    [[nodiscard]] LayoutTransposeMode parse_layout_transpose_mode(std::string_view raw, std::string_view option_name)
    {
        const std::string lowered = to_lower_ascii(raw);
        if (lowered == "auto")
        {
            return LayoutTransposeMode::auto_select;
        }
        if (lowered == "on")
        {
            return LayoutTransposeMode::force_enable;
        }
        if (lowered == "off")
        {
            return LayoutTransposeMode::force_disable;
        }

        throw CliArgumentError("Invalid value for --" + std::string(option_name) +
                               ": expected one of {auto, on, off}.");
    }

    /**
     * @brief 将文件格式转为字符串；Convert file format to string.
     * @param format 格式枚举；Format enum.
     * @return 字符串表示；String representation.
     */
    [[nodiscard]] std::string_view matrix_format_to_string(pipeline::MatrixFileFormat format)
    {
        switch (format)
        {
        case pipeline::MatrixFileFormat::auto_detect:
            return "auto";
        case pipeline::MatrixFileFormat::mat:
            return "mat";
        case pipeline::MatrixFileFormat::txt:
            return "txt";
        }

        return "unknown";
    }

    /**
     * @brief 将布局转置策略转为字符串；Convert layout-transpose mode to string.
     * @param mode 布局转置策略；Layout-transpose mode.
     * @return 字符串表示；String representation.
     */
    [[nodiscard]] std::string_view layout_transpose_mode_to_string(LayoutTransposeMode mode)
    {
        switch (mode)
        {
        case LayoutTransposeMode::auto_select:
            return "auto";
        case LayoutTransposeMode::force_enable:
            return "on";
        case LayoutTransposeMode::force_disable:
            return "off";
        }
        return "unknown";
    }

    /**
     * @brief 由文件扩展名推断矩阵格式；Infer matrix format from file extension.
     * @param path 文件路径；File path.
     * @return 若可识别返回格式，否则返回空；Returns format when recognized, otherwise empty.
     */
    [[nodiscard]] std::optional<pipeline::MatrixFileFormat> detect_matrix_format_from_extension(
        const std::filesystem::path &path)
    {
        const std::string lowered = to_lower_ascii(path.extension().string());
        if (lowered == ".mat")
        {
            return pipeline::MatrixFileFormat::mat;
        }
        if (lowered == ".txt")
        {
            return pipeline::MatrixFileFormat::txt;
        }
        return std::nullopt;
    }

    /**
     * @brief 获取格式对应默认扩展名；Get canonical extension of one format.
     * @param format 目标格式；Target format.
     * @return 扩展名字符串；File extension string.
     */
    [[nodiscard]] std::string canonical_extension_for_format(pipeline::MatrixFileFormat format)
    {
        if (format == pipeline::MatrixFileFormat::txt)
        {
            return ".txt";
        }
        return ".mat";
    }

    /**
     * @brief 解析正整数；Parse positive integer.
     * @param raw 文本值；Raw value text.
     * @param option_name 选项名；Option name.
     * @return 解析结果；Parsed integer.
     */
    [[nodiscard]] int parse_positive_int(std::string_view raw, std::string_view option_name)
    {
        const std::string text(raw);
        std::size_t parsed = 0;
        int value = 0;
        try
        {
            value = std::stoi(text, &parsed, 10);
        }
        catch (const std::exception &)
        {
            throw CliArgumentError("Invalid integer value for --" + std::string(option_name) + ".");
        }

        if (parsed != text.size())
        {
            throw CliArgumentError("Invalid integer value for --" + std::string(option_name) + ".");
        }
        if (value <= 0)
        {
            throw CliArgumentError("Option --" + std::string(option_name) + " must be positive.");
        }
        return value;
    }

    /**
     * @brief 解析正整数（size_t）；Parse positive size_t integer.
     * @param raw 文本值；Raw value text.
     * @param option_name 选项名；Option name.
     * @return 解析结果；Parsed size.
     */
    [[nodiscard]] std::size_t parse_positive_size(std::string_view raw, std::string_view option_name)
    {
        const std::string text(raw);
        std::size_t parsed = 0;
        unsigned long long value = 0ULL;
        try
        {
            value = std::stoull(text, &parsed, 10);
        }
        catch (const std::exception &)
        {
            throw CliArgumentError("Invalid integer value for --" + std::string(option_name) + ".");
        }

        if (parsed != text.size())
        {
            throw CliArgumentError("Invalid integer value for --" + std::string(option_name) + ".");
        }
        if (value == 0ULL)
        {
            throw CliArgumentError("Option --" + std::string(option_name) + " must be positive.");
        }
        return static_cast<std::size_t>(value);
    }

    /**
     * @brief 解析正浮点数；Parse positive floating-point value.
     * @param raw 文本值；Raw value text.
     * @param option_name 选项名；Option name.
     * @return 解析结果；Parsed floating-point value.
     */
    [[nodiscard]] double parse_positive_double(std::string_view raw, std::string_view option_name)
    {
        const std::string text(raw);
        std::size_t parsed = 0;
        double value = 0.0;
        try
        {
            value = std::stod(text, &parsed);
        }
        catch (const std::exception &)
        {
            throw CliArgumentError("Invalid floating-point value for --" + std::string(option_name) + ".");
        }

        if (parsed != text.size())
        {
            throw CliArgumentError("Invalid floating-point value for --" + std::string(option_name) + ".");
        }
        if (!std::isfinite(value) || value <= 0.0)
        {
            throw CliArgumentError("Option --" + std::string(option_name) + " must be a positive finite number.");
        }
        return value;
    }

    /**
     * @brief 规范化路径用于比较；Normalize path for equality comparison.
     * @param path 输入路径；Input path.
     * @return 规范化路径；Normalized path.
     */
    [[nodiscard]] std::filesystem::path normalized_path_for_compare(const std::filesystem::path &path)
    {
        std::error_code error;
        const std::filesystem::path absolute = std::filesystem::absolute(path, error);
        if (error)
        {
            return path.lexically_normal();
        }
        return absolute.lexically_normal();
    }

    /**
     * @brief 规范化运行选项；Normalize run-time options.
     * @param options CLI 选项（原地修改）；CLI options (modified in-place).
     */
    void normalize_run_options(CliOptions &options)
    {
        if (options.input_path.empty())
        {
            throw CliArgumentError("Missing input file. Use --input <path>.");
        }

        const std::optional<pipeline::MatrixFileFormat> input_ext_format =
            detect_matrix_format_from_extension(options.input_path);
        if (options.input_format == pipeline::MatrixFileFormat::auto_detect && !input_ext_format.has_value())
        {
            throw CliArgumentError("Cannot infer input format from extension. Use --input-format {mat|txt}.");
        }

        if (options.output_path.empty())
        {
            pipeline::MatrixFileFormat output_format = options.output_format;
            if (output_format == pipeline::MatrixFileFormat::auto_detect)
            {
                if (options.input_format != pipeline::MatrixFileFormat::auto_detect)
                {
                    output_format = options.input_format;
                }
                else if (input_ext_format.has_value())
                {
                    output_format = input_ext_format.value();
                }
                else
                {
                    output_format = pipeline::MatrixFileFormat::mat;
                }
                options.output_format = output_format;
            }

            std::string stem = options.input_path.stem().string();
            if (stem.empty())
            {
                stem = "result";
            }
            options.output_path = options.input_path.parent_path() /
                                  (stem + ".svd" + canonical_extension_for_format(output_format));
        }

        const std::optional<pipeline::MatrixFileFormat> output_ext_format =
            detect_matrix_format_from_extension(options.output_path);
        if (options.output_format == pipeline::MatrixFileFormat::auto_detect && !output_ext_format.has_value())
        {
            pipeline::MatrixFileFormat fallback = pipeline::MatrixFileFormat::mat;
            if (options.input_format != pipeline::MatrixFileFormat::auto_detect)
            {
                fallback = options.input_format;
            }
            else if (input_ext_format.has_value())
            {
                fallback = input_ext_format.value();
            }

            if (options.output_path.extension().empty())
            {
                options.output_path += canonical_extension_for_format(fallback);
            }
            options.output_format = fallback;
        }
    }

    /**
     * @brief 校验运行选项；Validate run-time options.
     * @param options CLI 选项；CLI options.
     */
    void validate_run_options(const CliOptions &options)
    {
        if (!std::filesystem::exists(options.input_path))
        {
            throw CliArgumentError("Input file does not exist: " + options.input_path.string());
        }
        if (std::filesystem::is_directory(options.input_path))
        {
            throw CliArgumentError("Input path is a directory, expected a file: " + options.input_path.string());
        }
        if (std::filesystem::is_directory(options.output_path))
        {
            throw CliArgumentError("Output path points to a directory: " + options.output_path.string());
        }

        const std::filesystem::path lhs = normalized_path_for_compare(options.input_path);
        const std::filesystem::path rhs = normalized_path_for_compare(options.output_path);
        if (lhs == rhs)
        {
            throw CliArgumentError("Input and output paths must be different files.");
        }

        if (std::filesystem::exists(options.output_path) && !options.force_overwrite)
        {
            throw CliArgumentError("Output file already exists. Use --force to overwrite: " +
                                   options.output_path.string());
        }
    }

    /**
     * @brief 将 CLI 选项转换为 Pipeline 配置；Convert CLI options to pipeline config.
     * @param options CLI 选项；CLI options.
     * @return Pipeline 配置；Pipeline configuration.
     */
    [[nodiscard]] pipeline::PipelineConfig make_pipeline_config(const CliOptions &options)
    {
        pipeline::PipelineConfig config{};
        config.input_path = options.input_path;
        config.output_path = options.output_path;
        config.input_format = options.input_format;
        config.output_format = options.output_format;
        config.max_queued_results = options.queue_capacity;
        config.kernel_config = JacobiSvdConfig{
            .epsilon = options.epsilon,
            .max_sweeps = options.max_sweeps,
            .threads_per_block = options.threads_per_block,
            .layout_transpose_mode = options.layout_transpose_mode,
            .layout_transpose_min_columns = options.layout_transpose_min_columns,
            .layout_transpose_min_elements = options.layout_transpose_min_elements,
            .layout_transpose_auto_tune = options.layout_transpose_auto_tune,
            .layout_transpose_benchmark_repetitions = options.layout_transpose_benchmark_repetitions,
            .layout_transpose_benchmark_sweeps = options.layout_transpose_benchmark_sweeps,
        };
        return config;
    }

    /**
     * @brief 输出 dry-run 配置摘要；Print dry-run configuration summary.
     * @param options CLI 选项；CLI options.
     */
    void print_dry_run_config(const CliOptions &options, bool include_dry_run_banner)
    {
        if (include_dry_run_banner)
        {
            std::cout << "Dry run: pipeline was not executed.\n";
        }
        std::cout << "input           : " << options.input_path.string() << '\n';
        std::cout << "output          : " << options.output_path.string() << '\n';
        std::cout << "input-format    : " << matrix_format_to_string(options.input_format) << '\n';
        std::cout << "output-format   : " << matrix_format_to_string(options.output_format) << '\n';
        std::cout << "epsilon         : " << options.epsilon << '\n';
        std::cout << "max-sweeps      : " << options.max_sweeps << '\n';
        std::cout << "threads-per-blk : " << options.threads_per_block << '\n';
        std::cout << "layout-mode     : " << layout_transpose_mode_to_string(options.layout_transpose_mode) << '\n';
        std::cout << "layout-min-cols : " << options.layout_transpose_min_columns << '\n';
        std::cout << "layout-min-elem : " << options.layout_transpose_min_elements << '\n';
        std::cout << "layout-auto-tune: " << (options.layout_transpose_auto_tune ? "true" : "false") << '\n';
        std::cout << "layout-bench-rep: " << options.layout_transpose_benchmark_repetitions << '\n';
        std::cout << "layout-bench-swp: " << options.layout_transpose_benchmark_sweeps << '\n';
        std::cout << "queue-capacity  : " << options.queue_capacity << '\n';
        std::cout << "force-overwrite : " << (options.force_overwrite ? "true" : "false") << '\n';
    }

    /**
     * @brief 输出文本执行报告；Print text execution report.
     * @param report Pipeline 报告；Pipeline report.
     */
    void print_text_report(const pipeline::PipelineReport &report, double elapsed_milliseconds)
    {
        std::cout << "Pipeline completed.\n";
        std::cout << "testcases       : " << report.testcase_count << '\n';
        std::cout << "emitted-matrices: " << report.emitted_matrix_count << '\n';
        std::cout << "total-sweeps    : " << report.total_sweeps << '\n';
        std::cout << "layout-mode     : " << layout_transpose_mode_to_string(report.layout_transpose_mode) << '\n';
        std::cout << "layout-min-cols : " << report.layout_transpose_min_columns << '\n';
        std::cout << "layout-min-elem : " << report.layout_transpose_min_elements << '\n';
        std::cout << "layout-auto-tune: " << (report.layout_transpose_auto_tuned ? "true" : "false") << '\n';
        std::cout << "layout-best-spd : " << std::fixed << std::setprecision(3)
                  << report.layout_transpose_estimated_best_speedup << '\n';
        std::cout << "elapsed-ms      : " << std::fixed << std::setprecision(3)
                  << elapsed_milliseconds << '\n';
    }

    /**
     * @brief 输出 JSON 执行报告；Print JSON execution report.
     * @param report Pipeline 报告；Pipeline report.
     */
    void print_json_report(const pipeline::PipelineReport &report, double elapsed_milliseconds)
    {
        std::cout << "{\n";
        std::cout << "  \"testcase_count\": " << report.testcase_count << ",\n";
        std::cout << "  \"emitted_matrix_count\": " << report.emitted_matrix_count << ",\n";
        std::cout << "  \"total_sweeps\": " << report.total_sweeps << ",\n";
        std::cout << "  \"layout_transpose_mode\": \""
                  << layout_transpose_mode_to_string(report.layout_transpose_mode) << "\",\n";
        std::cout << "  \"layout_transpose_min_columns\": " << report.layout_transpose_min_columns << ",\n";
        std::cout << "  \"layout_transpose_min_elements\": " << report.layout_transpose_min_elements << ",\n";
        std::cout << "  \"layout_transpose_auto_tuned\": "
                  << (report.layout_transpose_auto_tuned ? "true" : "false") << ",\n";
        std::cout << "  \"layout_transpose_estimated_best_speedup\": " << std::fixed << std::setprecision(3)
                  << report.layout_transpose_estimated_best_speedup << ",\n";
        std::cout << "  \"elapsed_ms\": " << std::fixed << std::setprecision(3)
                  << elapsed_milliseconds << '\n';
        std::cout << "}\n";
    }

    ArgParser::ArgParser()
        : definitions_{
              {OptionId::input, "input", 'i', true, "PATH", "Input matrix stream file path."},
              {OptionId::output, "output", 'o', true, "PATH", "Output matrix stream file path."},
              {OptionId::input_format, "input-format", '\0', true, "FMT", "Input format: auto|mat|txt."},
              {OptionId::output_format, "output-format", '\0', true, "FMT", "Output format: auto|mat|txt."},
              {OptionId::format, "format", 'f', true, "FMT", "Set both input/output format: auto|mat|txt."},
              {OptionId::epsilon, "epsilon", 'e', true, "NUM", "Convergence epsilon (>0)."},
              {OptionId::max_sweeps, "max-sweeps", 's', true, "N", "Maximum sweep count (>0)."},
              {OptionId::threads_per_block, "threads-per-block", 't', true, "N", "CUDA threads per block (>0)."},
              {OptionId::layout_transpose_mode, "layout-transpose-mode", '\0', true, "MODE",
               "Layout-transpose mode: auto|on|off."},
              {OptionId::layout_transpose_min_columns, "layout-transpose-min-cols", '\0', true, "N",
               "Auto-mode threshold: minimum columns (>0)."},
              {OptionId::layout_transpose_min_elements, "layout-transpose-min-elems", '\0', true, "N",
               "Auto-mode threshold: minimum elements (>0)."},
              {OptionId::layout_transpose_auto_tune, "layout-transpose-auto-tune", '\0', false, "",
               "Run micro-benchmark to auto-tune layout thresholds before pipeline."},
              {OptionId::layout_transpose_benchmark_repetitions, "layout-transpose-bench-reps", '\0', true, "N",
               "Auto-tune repetitions per scanned matrix size (>0)."},
              {OptionId::layout_transpose_benchmark_sweeps, "layout-transpose-bench-sweeps", '\0', true, "N",
               "Auto-tune sweep cap per benchmark run (>0)."},
              {OptionId::queue_capacity, "queue-capacity", 'c', true, "N", "In-flight/reorder window size (>0)."},
              {OptionId::force, "force", 'y', false, "", "Overwrite existing output file."},
              {OptionId::dry_run, "dry-run", '\0', false, "", "Validate arguments and print config only."},
              {OptionId::print_config, "print-config", '\0', false, "", "Print effective config before execution."},
              {OptionId::json_report, "json-report", '\0', false, "", "Print execution report in JSON."},
              {OptionId::quiet, "quiet", 'q', false, "", "Suppress text report."},
              {OptionId::help, "help", 'h', false, "", "Show this help message."},
              {OptionId::version, "version", 'v', false, "", "Show version information."},
          }
    {
    }

    ParseResult ArgParser::parse(int argc, char *const argv[]) const
    {
        ParseResult result{};
        std::vector<std::string_view> positional_arguments;

        std::size_t index = 1;
        while (index < static_cast<std::size_t>(argc))
        {
            const std::string_view token(argv[index] == nullptr ? "" : argv[index]);
            if (token.empty())
            {
                ++index;
                continue;
            }

            if (token == "--")
            {
                ++index;
                while (index < static_cast<std::size_t>(argc))
                {
                    positional_arguments.emplace_back(argv[index] == nullptr ? "" : argv[index]);
                    ++index;
                }
                break;
            }

            if (token.rfind("--", 0) == 0)
            {
                parse_long_option(token, argv, argc, index, result);
                ++index;
                continue;
            }

            if (token.size() > 1 && token[0] == '-')
            {
                parse_short_options(token, argv, argc, index, result);
                ++index;
                continue;
            }

            positional_arguments.push_back(token);
            ++index;
        }

        std::size_t positional_index = 0;
        if (result.options.input_path.empty() && positional_index < positional_arguments.size())
        {
            result.options.input_path = std::filesystem::path(std::string(positional_arguments[positional_index]));
            ++positional_index;
        }
        if (result.options.output_path.empty() && positional_index < positional_arguments.size())
        {
            result.options.output_path = std::filesystem::path(std::string(positional_arguments[positional_index]));
            ++positional_index;
        }
        if (positional_index < positional_arguments.size())
        {
            throw CliArgumentError("Unexpected positional argument: " +
                                   std::string(positional_arguments[positional_index]));
        }

        if (result.action == ParseAction::run)
        {
            normalize_run_options(result.options);
            validate_run_options(result.options);
        }

        return result;
    }

    std::string ArgParser::help_message(std::string_view executable) const
    {
        std::ostringstream stream;
        stream << "Jacobi SVD CUDA CLI\n\n";
        stream << "Usage:\n";
        stream << "  " << executable << " [OPTIONS] <input> [output]\n\n";
        stream << "Options:\n";

        for (const OptionDefinition &option : definitions_)
        {
            std::ostringstream names;
            if (option.short_name != '\0')
            {
                names << '-' << option.short_name << ", ";
            }
            else
            {
                names << "    ";
            }

            names << "--" << option.long_name;
            if (option.requires_value)
            {
                names << ' ' << option.value_hint;
            }

            stream << "  " << names.str();
            const std::size_t padding = names.str().size() < 32U ? (32U - names.str().size()) : 1U;
            stream << std::string(padding, ' ');
            stream << option.description << '\n';
        }

        stream << "\nExamples:\n";
        stream << "  " << executable << " -i experiments/inputs/a.mat -o experiments/outputs/r.mat\n";
        stream << "  " << executable << " experiments/inputs/a.mat --print-config\n";
        stream << "  " << executable << " input.txt output.txt --format txt --epsilon 1e-10\n";
        stream << "  " << executable << " --input a.mat --output b.txt --output-format txt --json-report --force\n";
        stream << "  " << executable
               << " --input a.mat --layout-transpose-auto-tune --layout-transpose-mode auto\n";
        stream << "\nNotes:\n";
        stream << "  - When [output] is omitted, default output is <input-stem>.svd.{mat|txt}.\n";
        stream << "  - Existing output file requires --force to overwrite.\n";
        return stream.str();
    }

    void ArgParser::parse_long_option(std::string_view token,
                                      char *const argv[],
                                      int argc,
                                      std::size_t &index,
                                      ParseResult &result) const
    {
        const std::string_view body = token.substr(2);
        if (body.empty())
        {
            throw CliArgumentError("Invalid option token '--'.");
        }

        const std::size_t eq_pos = body.find('=');
        const std::string_view option_name = (eq_pos == std::string_view::npos) ? body : body.substr(0, eq_pos);
        const bool has_inline_value = (eq_pos != std::string_view::npos);
        const std::string_view inline_value = has_inline_value ? body.substr(eq_pos + 1) : std::string_view{};

        const OptionDefinition *option = find_long_option(option_name);
        if (option == nullptr)
        {
            throw CliArgumentError("Unknown option: --" + std::string(option_name));
        }

        if (option->requires_value)
        {
            const std::string_view value =
                has_inline_value ? inline_value : consume_next_value(option->long_name, argv, argc, index);
            apply_option(*option, value, result);
            return;
        }

        if (has_inline_value)
        {
            throw CliArgumentError("Option --" + std::string(option->long_name) + " does not take a value.");
        }
        apply_option(*option, std::nullopt, result);
    }

    void ArgParser::parse_short_options(std::string_view token,
                                        char *const argv[],
                                        int argc,
                                        std::size_t &index,
                                        ParseResult &result) const
    {
        if (token.size() <= 1)
        {
            throw CliArgumentError("Invalid short option token.");
        }

        std::size_t offset = 1;
        while (offset < token.size())
        {
            const char short_name = token[offset];
            const OptionDefinition *option = find_short_option(short_name);
            if (option == nullptr)
            {
                throw CliArgumentError(std::string("Unknown short option: -") + short_name);
            }

            if (!option->requires_value)
            {
                apply_option(*option, std::nullopt, result);
                ++offset;
                continue;
            }

            std::string_view value;
            if (offset + 1 < token.size())
            {
                value = token.substr(offset + 1);
            }
            else
            {
                value = consume_next_value(option->long_name, argv, argc, index);
            }

            apply_option(*option, value, result);
            break;
        }
    }

    void ArgParser::apply_option(const OptionDefinition &option,
                                 const std::optional<std::string_view> &value,
                                 ParseResult &result) const
    {
        switch (option.id)
        {
        case OptionId::input:
            result.options.input_path = std::filesystem::path(std::string(value.value()));
            return;
        case OptionId::output:
            result.options.output_path = std::filesystem::path(std::string(value.value()));
            return;
        case OptionId::input_format:
            result.options.input_format = parse_matrix_format(value.value(), option.long_name);
            return;
        case OptionId::output_format:
            result.options.output_format = parse_matrix_format(value.value(), option.long_name);
            return;
        case OptionId::format:
        {
            const pipeline::MatrixFileFormat format = parse_matrix_format(value.value(), option.long_name);
            result.options.input_format = format;
            result.options.output_format = format;
            return;
        }
        case OptionId::epsilon:
            result.options.epsilon = parse_positive_double(value.value(), option.long_name);
            return;
        case OptionId::max_sweeps:
            result.options.max_sweeps = parse_positive_int(value.value(), option.long_name);
            return;
        case OptionId::threads_per_block:
            result.options.threads_per_block = parse_positive_int(value.value(), option.long_name);
            return;
        case OptionId::layout_transpose_mode:
            result.options.layout_transpose_mode = parse_layout_transpose_mode(value.value(), option.long_name);
            return;
        case OptionId::layout_transpose_min_columns:
            result.options.layout_transpose_min_columns = parse_positive_int(value.value(), option.long_name);
            return;
        case OptionId::layout_transpose_min_elements:
            result.options.layout_transpose_min_elements = parse_positive_size(value.value(), option.long_name);
            return;
        case OptionId::layout_transpose_auto_tune:
            result.options.layout_transpose_auto_tune = true;
            return;
        case OptionId::layout_transpose_benchmark_repetitions:
            result.options.layout_transpose_benchmark_repetitions = parse_positive_int(value.value(), option.long_name);
            return;
        case OptionId::layout_transpose_benchmark_sweeps:
            result.options.layout_transpose_benchmark_sweeps = parse_positive_int(value.value(), option.long_name);
            return;
        case OptionId::queue_capacity:
            result.options.queue_capacity = parse_positive_size(value.value(), option.long_name);
            return;
        case OptionId::force:
            result.options.force_overwrite = true;
            return;
        case OptionId::dry_run:
            result.options.dry_run = true;
            return;
        case OptionId::print_config:
            result.options.print_config = true;
            return;
        case OptionId::json_report:
            result.options.json_report = true;
            return;
        case OptionId::quiet:
            result.options.quiet = true;
            return;
        case OptionId::help:
            result.action = ParseAction::help;
            return;
        case OptionId::version:
            result.action = ParseAction::version;
            return;
        }
    }

    const OptionDefinition *ArgParser::find_long_option(std::string_view name) const
    {
        const auto iterator = std::find_if(definitions_.begin(),
                                           definitions_.end(),
                                           [name](const OptionDefinition &option) {
                                               return option.long_name == name;
                                           });
        if (iterator == definitions_.end())
        {
            return nullptr;
        }
        return &(*iterator);
    }

    const OptionDefinition *ArgParser::find_short_option(char name) const
    {
        const auto iterator = std::find_if(definitions_.begin(),
                                           definitions_.end(),
                                           [name](const OptionDefinition &option) {
                                               return option.short_name == name;
                                           });
        if (iterator == definitions_.end())
        {
            return nullptr;
        }
        return &(*iterator);
    }

    std::string_view ArgParser::consume_next_value(std::string_view option_name,
                                                   char *const argv[],
                                                   int argc,
                                                   std::size_t &index) const
    {
        const std::size_t next_index = index + 1;
        if (next_index >= static_cast<std::size_t>(argc) || argv[next_index] == nullptr)
        {
            throw CliArgumentError("Option --" + std::string(option_name) + " requires a value.");
        }

        index = next_index;
        return std::string_view(argv[index]);
    }
} // namespace jacobi::svd::cli

/**
 * @brief 程序入口；Program entry point.
 * @param argc 参数个数；Argument count.
 * @param argv 参数数组；Argument vector.
 * @return 进程退出码；Process exit code.
 */
int main(int argc, char *argv[])
{
    try
    {
        const std::string executable =
            (argc > 0 && argv[0] != nullptr) ? std::filesystem::path(argv[0]).filename().string() : "jacobi-svd";

        const jacobi::svd::cli::ArgParser parser{};
        const jacobi::svd::cli::ParseResult parsed = parser.parse(argc, argv);

        if (parsed.action == jacobi::svd::cli::ParseAction::help)
        {
            std::cout << parser.help_message(executable);
            return 0;
        }
        if (parsed.action == jacobi::svd::cli::ParseAction::version)
        {
            std::cout << "jacobi-svd-cuda CLI v0.1.0\n";
            return 0;
        }

        if (parsed.options.dry_run)
        {
            jacobi::svd::cli::print_dry_run_config(parsed.options, true);
            return 0;
        }

        if (parsed.options.print_config)
        {
            std::cout << "Effective configuration:\n";
            jacobi::svd::cli::print_dry_run_config(parsed.options, false);
            std::cout << '\n';
        }

        const jacobi::svd::pipeline::PipelineConfig config =
            jacobi::svd::cli::make_pipeline_config(parsed.options);
        const auto started_at = std::chrono::steady_clock::now();
        const jacobi::svd::pipeline::PipelineReport report = jacobi::svd::pipeline::run_pipeline(config);
        const auto finished_at = std::chrono::steady_clock::now();
        const double elapsed_milliseconds =
            std::chrono::duration<double, std::milli>(finished_at - started_at).count();

        if (!parsed.options.quiet)
        {
            jacobi::svd::cli::print_text_report(report, elapsed_milliseconds);
        }
        if (parsed.options.json_report)
        {
            jacobi::svd::cli::print_json_report(report, elapsed_milliseconds);
        }

        return 0;
    }
    catch (const jacobi::svd::cli::CliArgumentError &error)
    {
        std::cerr << "Argument error: " << error.what() << '\n';
        std::cerr << "Use --help for usage.\n";
        return 2;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Execution failed: " << error.what() << '\n';
        return 1;
    }
}
