## marimoの主な機能と特徴

公式ドキュメントの内容を引用

> ⚡️ リアクティブ実行：依存セルが自動更新される  
> 🖐️ インタラクティブUI：スライダー・テーブル・チャートなどを簡単に追加  
> 🐍 Gitフレンドリー：ノートブックは.py形式で保存  
> 🛢️ SQL対応：Python変数を使ったSQLクエリが可能  
> 🤖 AI補完：データに特化したAIアシスタントでコード生成  
> 🔬 再現性重視：隠れた状態なし、決定論的な実行順序  
> 🛜 Webアプリ化：ノートブックをそのままWebアプリやスライドに変換  
> 🧪 テスト可能：pytestでノートブックをテスト  
> ⌨️ モダンエディタ：Copilot、Vim操作、変数ビューアなどを搭載

### marimoのインストール

```bash
pip install marimo
uv add marimo　#uv経由
```

### チュートリアルを起動する

```bash
marimo tutorial intro
uv run marimo tutorial intoro  #uv経由
```

marimoのチュートリアル画面が表示される。

### ノートブックを新規作成・編集する

#### 新規作成（空のノートブック）

```bash
marimo edit my_notebook.py
uv run marimo edit my_notebook.py  #uv経由
```

このコマンドで `my_notebook.py` というPythonファイルが作成され、marimoのエディタが起動する。

#### 既存ノートブックの編集

すでにある `.py` ファイルを編集したい場合も同様に

```bash
marimo edit existing_notebook.py
uv run marimo edit existing_notebook.py  #uv経由
```

### ノートブックをWebアプリとして実行

作成したノートブックをアプリとして起動するには、

```bash
marimo run my_notebook.py
uv run marimo run my_notebook.py　　#uv経由
```

この状態では、コードは非表示になり、UIだけが表示されるWebアプリとして動作する。

出典: https://zenn.dev/dxc_ai_driven/articles/10e04000ff6f1d
