#!/bin/bash
# fix_git.sh - 安全清理仓库
set -e

echo "[1/3] 正在从 Git 缓存中移除 .venv (文件保留)..."
git rm -r --cached .venv --ignore-unmatch

echo "[2/3] 创建标准的 .gitignore..."
cat > .gitignore <<EOF
.venv/
data/
model/
__pycache__/
.DS_Store
EOF

echo "[3/3] 提交更改..."
git add .gitignore
git commit -m "chore: stop tracking venv and add gitignore"

echo "完成！现在你可以尝试 git push 了。"
