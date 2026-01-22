# CLAUDE.md

## プロジェクト概要

Kaggle LLM Book のサンプルコード・notebookを管理するリポジトリ

## Kaggle Notebook + GitHub連携のワークフロー

### 基本フロー

```
ローカル編集 → GitHub push → Kaggleでインポート → Kaggle上で実行
```

### 重要: データパスの設定

**Kaggleでデータセットを追加した場合、マウントパスを必ず確認すること**

```python
# Kaggle notebook上でパスを確認
import os
print(os.listdir("/kaggle/input/"))
```

- データセット名とマウントパスは**一致しないことがある**
- 例: `takaito/atmacup17` を追加しても `/kaggle/input/atmacup17/` ではなく `/kaggle/input/` 直下にマウントされる場合がある

### notebookのダウンロード

```bash
kaggle kernels pull <username>/<kernel-slug> -m
```

### notebookのpush（Kaggle API経由）

```bash
kaggle kernels push
```

## 実行結果をClaude Codeと共有する方法

### 方法1: Kaggle APIで出力ダウンロード（推奨）

**前提**: notebookを **Save & Run All (Commit)** した後

```bash
kaggle kernels output <username>/<kernel-slug> -p ./output
```

出力されたログファイルを Claude Code が読み取って結果を確認できる。

### 方法2: スクリーンショット

スクリーンショットを撮ってパスを共有:
```
c:\Users\hikar\OneDrive\画像\Screenshots\スクリーンショット XXXX.png
```

### 方法3: 実行済みnotebookをpull

Commit後、出力付きのnotebookをダウンロード:
```bash
kaggle kernels pull <username>/<kernel-slug>
```

## ディレクトリ構造

```
kaggle_llm_book/
├── notebooks/          # Kaggle notebookを格納
│   └── *.ipynb
├── README.md
└── CLAUDE.md
```
