#!/usr/bin/env bash

# 一键完成环境检查、增量编译和正确性测试。
#
# 直接执行虚拟环境里的 python，并不会自动把该虚拟环境的 bin 目录
# 加入 PATH。PyTorch BuildExtension 是通过 PATH 查找 ninja 的；因此这里
# 显式补齐 PATH，避免悄悄回退到速度较慢的 distutils 编译后端。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 可以通过 AI_INFRA_PYTHON 指定其他 Python；默认优先使用已激活的
# virtualenv，其次使用本项目当前机器上的 ~/venvs/torch。
if [[ -n "${AI_INFRA_PYTHON:-}" ]]; then
    PYTHON_BIN="${AI_INFRA_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${HOME}/venvs/torch/bin/python" ]]; then
    PYTHON_BIN="${HOME}/venvs/torch/bin/python"
else
    PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "错误：没有找到可执行的 Python。请设置 AI_INFRA_PYTHON=/path/to/python。" >&2
    exit 1
fi

# sys.executable 能保留虚拟环境路径。这里不能用 realpath：venv 的 python
# 通常是符号链接，解析后会错误地回到 /usr/bin，进而再次找不到 venv 中的 ninja。
PYTHON_BIN_DIR="$("${PYTHON_BIN}" -c 'import os, sys; print(os.path.dirname(os.path.abspath(sys.executable)))')"
export PATH="${PYTHON_BIN_DIR}:/usr/local/cuda/bin:${PATH}"

if ! command -v ninja >/dev/null 2>&1; then
    echo "错误：当前 Python 环境没有可用的 ninja。" >&2
    echo "请运行：${PYTHON_BIN} -m pip install ninja" >&2
    exit 1
fi

NINJA_VERSION="$(ninja --version)"
# Ninja 1.13.2 存在已知的依赖数据库回归：.ninja_deps 会持续报告
# "premature end of file"，随后把所有目标误判为需要重编。PyPI 尚无
# 含修复的 1.14 时，固定使用已验证的 1.11.1.4。
if [[ "${NINJA_VERSION}" == 1.13.2* ]]; then
    echo "错误：检测到 Ninja ${NINJA_VERSION}，该版本会导致本项目重复全量编译。" >&2
    echo "请运行：${PYTHON_BIN} -m pip install --upgrade ninja==1.11.1.4" >&2
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "错误：PATH 中没有 nvcc，请检查 CUDA Toolkit 是否安装在 /usr/local/cuda。" >&2
    exit 1
fi

# 编译和测试都依赖可用的 CUDA GPU。提前检查能给出比 nvcc/linker
# 报错更直接的信息，并从当前 GPU 自动推导目标架构（RTX 3060 为 8.6）。
GPU_ARCH="$("${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "错误：torch.cuda.is_available() == False；请先检查 WSL GPU 映射、"
        "NVIDIA 驱动以及 PyTorch CUDA 版本。"
    )

major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${GPU_ARCH}}"
# 6GB 笔记本环境中并行启动多个 nvcc 容易耗尽 WSL 内存；需要时可由
# 调用者显式覆盖，例如 MAX_JOBS=2 ./build_and_test.sh。
export MAX_JOBS="${MAX_JOBS:-1}"

cd "${SCRIPT_DIR}"

echo "Python : ${PYTHON_BIN}"
echo "Ninja  : $(command -v ninja) (${NINJA_VERSION})"
echo "NVCC   : $(command -v nvcc)"
echo "Arch   : ${TORCH_CUDA_ARCH_LIST}"
echo "Jobs   : ${MAX_JOBS}"

echo "[1/2] 增量编译 ai_infra_ops"
"${PYTHON_BIN}" setup.py build_ext --inplace

echo "[2/2] 运行完整正确性测试"
"${PYTHON_BIN}" test_all.py

echo "完成：编译与正确性测试均通过。"
