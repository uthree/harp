# アーキテクチャ概要

## プロジェクト構成

```
src/
├── ast/          # 抽象構文木（カーネル表現）
├── graph/        # 計算グラフ（DAG）
│   └── shape/    # シェイプ式、メモリレイアウト
├── tensor/       # 高レベルTensor API
├── grad/         # 自動微分
├── lowerer/      # Graph→AST変換
├── opt/          # AST最適化
├── backend/      # バックエンド共通インターフェース
└── lib.rs        # 公開API

crates/
├── backend-c/        # C言語バックエンド
├── backend-cuda/     # CUDAバックエンド
├── backend-metal/    # Metalバックエンド
├── backend-opencl/   # OpenCLバックエンド
├── backend-openmp/   # OpenMPバックエンド
├── backend-rust/     # Rustバックエンド
├── dsl/              # DSLパーサー
├── nn/               # ニューラルネットワーク層
├── cli/              # CLIツール
└── viz/              # 可視化ツール
```

## データフロー

```
Tensor API (ユーザーコード)
    ↓
計算グラフ (GraphNode DAG)
    ↓ [グラフレベル最適化: 融合、冗長除去]
    ↓
Lowerer (Graph → AST)
    • ビュー融合
    • インデックス式生成
    • ループ生成
    ↓
AST (AstNode::Program)
    ↓ [AST最適化: ループ変換、並列化]
    ↓
バックエンドレンダラー (AST → コード)
    ↓
ターゲットコード (C/CUDA/Metal等)
    ↓
コンパイラ (nvcc, clang等)
    ↓
実行可能カーネル
    ↓
実行
```

## 主要コンポーネント

### 計算グラフ (graph/)

高レベルの計算を有向非巡回グラフ（DAG）で表現。ノードは演算、エッジはデータ依存関係を表す。

- `GraphNode`: 参照カウント付き計算ノード
- `GraphOp`: 演算種別（View, MapReduce, Unfold, Scatter, Scan）
- `View`: メモリレイアウト（ストライド or インデックス式）
- `Expr`: シンボリックなシェイプ/ストライド式

### AST (ast/)

バックエンド非依存のカーネル表現。ループ、分岐、メモリアクセス等を含む。

- `AstNode`: 計算ノード（算術、制御フロー、メモリ、関数）
- `DType`: データ型（F32, F64, I32等）
- `Literal`: リテラル値
- `ParallelInfo`: 並列化メタデータ

### Lowerer (lowerer/)

計算グラフをASTカーネルに変換。

- 演算融合（View融合、要素演算+リダクション融合）
- ViewからExprへのインデックス式生成
- Exprからループ構造の生成

### 最適化 (opt/)

ASTレベルの最適化パス。

- 探索戦略: ビームサーチ、枝刈りDFS/BFS、ルールベース
- サジェスター: ループ融合/タイリング/交換、並列化、ベクトル化、CSE
- コスト推定: 演算カウント + メモリアクセスモデル

### バックエンド (backend/)

プラグイン可能なコード生成・実行基盤。

- `Device`: デバイス抽象化
- `Buffer`: メモリバッファ
- `Compiler`: コンパイラ
- `Renderer`: コード生成
- `Kernel`: 実行可能カーネル

## 設計原則

1. **遅延評価**: テンソルは実行まで計算グラフのまま
2. **型安全**: コンパイル時の次元チェック（`Tensor<D, T>`）
3. **参照カウント**: `Rc<GraphInner>`によるDAG共有
4. **内部可変性**: `RefCell`によるバッファキャッシュ
5. **二段階AST**: Graph（高レベル）→ AST（実装レベル）
6. **多段階最適化**: グラフ融合 → AST変換 → バックエンド固有
