# TeX Structure

- `main.tex`: 文档入口（仅负责结构组织）
- `preamble.tex`: 导言区（包管理、页面样式、代码 listing）
- `sections/`: 分章节内容（当前 7 章）
- `figures/`: 图像资源
- `tables/`: 表格资源
- `latexmkrc`: 构建配置（输出在 `tex/build/`）

## Build

在 `tex/` 目录执行：

```powershell
latexmk -xelatex main.tex
```

清理：

```powershell
latexmk -c
```
