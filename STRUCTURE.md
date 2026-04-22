我们使用 C++20、CUDA 12.8，以 libcu++ 实现单边雅可比奇异值分解。

项目结构遵循领域驱动设计思想：

```
jacobi-svd-cuda/
├── CMakeLists.txt
│
├── experiments/
│   ├── inputs
│   └── outputs
│ 
├── tests/
│   └── ...
│ 
├── build/
│ 
├── tex/
│   ├── build/
│   ├── latexmkrc
│   └── ...
│
├── src/
│   ├── io.cu # 从文件中读矩阵、将矩阵写到文件中的基础设施
│   ├── pipeline.cpp # 将领域对象组装为应用层逻辑
│   ├── main.cpp # 入口，CLI 表示层逻辑
│   └── kernels.cu # 领域层 CUDA kernels
│
└── include/
    ├── io.hpp # 从文件中读矩阵、将矩阵写到文件中的基础设施
    ├── pipeline.hpp # 将领域对象组装为应用层逻辑
    └── kernels.hpp # 领域层 CUDA kernels
```

我们设计一种文件格式：*.mat 来二进制地保存矩阵的数值。文件由两部分交替组成：元数据和矩阵

```cpp
struct MatMetaData
{
    uint64_t rows;
    uint64_t columns;
};
```

使用网络字节序，矩阵数值使用 `double` 储存。`io.cpp`提供对 *.mat 的矩阵流抽象，在现代 C++ (C++20 及以上) 中，标准库引入了 <bit> 头文件中的 std::byteswap 和 std::endian，这为我们提供了极其优雅的跨平台解决方案。我们使用内存映射文件 (Memory-Mapped Files) 解决问题，`mmap()` (POSIX) / `CreateFileMapping` & `MapViewOfFile` & `FlushViewOfFile` (Windows)。我们定义一个 concept，确保传进来的 Policy 必须具备我们需要的能力。

`io.cpp` 还提供将 *.txt 抽象为矩阵流的函数：数字使用空格区分，行间使用`\n`区分。所以应该用模板元编程抽象出逻辑，将 *.mat 还是 *.txt 作为 Source policy 解耦进行 policy-based design：它们与如何存取字节流无关。读流和写流应当分别封装到两个类，不过也不影响解耦读写动作与序列化/反序列化的设计。

`pipeline.cpp`应用层抽象为：`testcases -> kernel -> output` 三个步骤的映射结合。这里采用“聚合根”的理念，将三个步骤的三个对象组合为一个对象。为避免线程写导致阻塞、统一写导致内存驻留，我们使用生产者-消费者模式组织 output 环节：它作为消费者将结果写出去；线程生命周期在类内部做 RAII 管理。接着我们采用重排序缓冲区，线程池来控制并发总数，主线程将一段一段的 tasks enqueue 到线程池队列的时候，就将 future 交给写的队列，然后写线程 get。这样结果有序，又是并行计算的。由于都是内存映射文件和调GPU，所以控制总量即可。

Reader 留在主线程，用单游标方案串行派发任务，顺便可以 `cudaMallocHost` 装载好，直接固定任务所使用的内存避免那个臭名昭著的硬性拷贝，然后把解析交给工作线程去做。`cudaMallocHost` 分析一下直接把结果的内存也预留，这样读写都在同一块只分配一次。

`main.cpp` 是 CLI 表示层的实现，将命令行参数组装 pipeline。

`kernels.cu` 算法伪代码如下（串行方式）：
```
// 算法：单边雅可比奇异值分解 (One-sided Jacobi SVD)
// 输入：实数矩阵 A (大小为 m x n)
// 输出：左奇异矩阵 U (m x n)，奇异值数组 Sigma (大小为 n)，右奇异矩阵 V (n x n)
// 参数：容差 epsilon (用于判断收敛，例如 1e-9)

function OneSidedJacobiSVD(A, epsilon):
    m, n = size(A)
    V = IdentityMatrix(n)  // 初始化 V 为 n x n 的单位矩阵
    
    converged = false
    sweeps = 0             // 记录扫尾(Sweep)的次数
    
    // 主循环：不断进行迭代，直到所有列正交
    while not converged:
        converged = true
        sweeps = sweeps + 1
        
        // 遍历所有可能的列对 (p, q) 进行正交化
        // 注意：在 CUDA 中，这层双重循环会被“巡回赛排序”和并行线程所替代
        for p = 1 to n - 1:
            for q = p + 1 to n:
                
                // 1. 计算两列的点积 (Inner Product) 与欧几里得范数平方 (Squared L2-Norm)
                a_pp = 0.0, a_qq = 0.0, a_pq = 0.0
                for i = 1 to m:
                    a_pp = a_pp + A[i, p] * A[i, p]
                    a_qq = a_qq + A[i, q] * A[i, q]
                    a_pq = a_pq + A[i, p] * A[i, q]
                
                // 2. 检查两列是否已经足够正交 (判断收敛条件)
                // 使用相对容差，避免矩阵数值整体极小或极大带来的误判
                if abs(a_pq) > epsilon * sqrt(a_pp * a_qq):
                    converged = false
                    
                    // 3. 计算稳定的 Givens 旋转角度 (Rutishauser 公式)
                    // 避免直接计算 arctan 导致的精度丢失
                    tau = (a_qq - a_pp) / (2.0 * a_pq)
                    
                    if tau >= 0:
                        t = 1.0 / (tau + sqrt(1.0 + tau * tau))
                    else:
                        t = -1.0 / (-tau + sqrt(1.0 + tau * tau))
                    
                    c = 1.0 / sqrt(1.0 + t * t)  // cos(theta)
                    s = t * c                    // sin(theta)
                    
                    // 4. 应用 Givens 旋转更新矩阵 A 的第 p 列和第 q 列
                    for i = 1 to m:
                        temp_Ap = A[i, p]
                        A[i, p] = c * temp_Ap - s * A[i, q]
                        A[i, q] = s * temp_Ap + c * A[i, q]
                    
                    // 5. 同样应用旋转更新右奇异矩阵 V 的第 p 列和第 q 列
                    for i = 1 to n:
                        temp_Vp = V[i, p]
                        V[i, p] = c * temp_Vp - s * V[i, q]
                        V[i, q] = s * temp_Vp + c * V[i, q]
    
    // --------------------------------------------------------------------------
    // 收敛后处理：提取奇异值 Sigma 和左奇异向量 U
    // 此时 A 已经变成了 U * Sigma
    // --------------------------------------------------------------------------
    U = AllocateMatrix(m, n)
    Sigma = AllocateArray(n)
    
    for j = 1 to n:
        // 计算每一列的 L2 范数，这就是奇异值
        norm_j = 0.0
        for i = 1 to m:
            norm_j = norm_j + A[i, j] * A[i, j]
        
        Sigma[j] = sqrt(norm_j)
        
        // 归一化 A 的列，得到左奇异向量
        for i = 1 to m:
            if Sigma[j] > epsilon:  // 防止除以零
                U[i, j] = A[i, j] / Sigma[j]
            else:
                U[i, j] = 0.0
                
    return U, Sigma, V
```

在 CUDA 这种高度并行的显卡架构里，如果你同时启动几千个线程去随心所欲地抽取矩阵的列来进行 Givens 旋转，一定会发生数据冲突。我们使用巡回赛排序 (Tournament Ordering)，将内外层循环扁平化，使得互相无冲突的列对能在不同的 CUDA Threads 中并行执行。它的核心思想是：固定一点，其余轮转 (Round-Robin with a Fixed Element)。

在 CUDA 中，直接用一个一维数组 (One-dimensional Array) 配合索引映射函数，是管理矩阵的最优解。因为它能保证物理内存的连续性，减少指针跳转。GPU 架构中，性能的命门在于内存合并访存 (Memory Coalescing)。当一个 Warp（32 个线程）同时发起内存请求时，如果这些线程访问的地址是连续的，显存控制器就能将这些请求合并为一个 Transaction（事务），带宽利用率拉满；如果地址是分散的（Strided Access），效率就会直线下降。

在手写的 Kernel 内部，为了那宝贵的带宽，请务必坚持行主序。而且 Matrix 应该用面向对象封装，多考虑 RAII 工具和 libcu++ 工具。当我们要调用 SVD 的 Kernel 函数时，我们不传递整个对象，而是只传递裸数据；但注意 kernel 必须设计为“纯函数”，而将管理逻辑交给封装。

但是我们思考一下单边雅可比算法，它是列主序的，但是我们的矩阵储存是行主序的。为了优化内存访问，布局转置应该在 GPU 上进行，所以你思考一下如何布置线程使得我们的布局转置能够并行化。在 GPU 上布局转置以后就开算，然后再转回来，注意避免分配不必要的内存。若矩阵很小或 sweep 很少，转置开销可能反而占主导。以考虑小矩阵不转置，因为实际上就是改变一个映射 policy 而已。
