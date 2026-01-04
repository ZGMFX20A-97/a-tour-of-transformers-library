import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from datasets import load_dataset, load_from_disk
    from transformers import AutoTokenizer
    return AutoTokenizer, load_dataset, load_from_disk


@app.cell
def _(mo):
    mo.md(r"""
    ### Datasetsの基本的な使い方
    """)
    return


@app.cell
def _(load_dataset):
    datasets = load_dataset("madao33/new-title-chinese")
    datasets
    return (datasets,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 複数のデータ集を含むDatasetsの使い方
    """)
    return


@app.cell
def _(load_dataset):
    boolq_datasets = load_dataset("super_glue", "boolq")
    boolq_datasets
    return (boolq_datasets,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Datasets内の種類に分けてロードする
    #### 訓練データ、検証データ、テストデータ
    """)
    return


@app.cell
def _(load_dataset):
    train_datasets = load_dataset("madao33/new-title-chinese", split="train")
    train_datasets
    return


@app.cell
def _(load_dataset):
    # 部分的に取得するとき(スライス)
    spliced_train_datasets = load_dataset(
        "madao33/new-title-chinese", split="train[:100]"
    )
    spliced_train_datasets
    return


@app.cell
def _(load_dataset):
    # 部分的に取得するとき(パーセンテージ)
    pct_train_datasets = load_dataset(
        "madao33/new-title-chinese", split="train[:50%]"
    )
    pct_train_datasets
    return


@app.cell
def _(load_dataset):
    # 同じ種類のデータセットでも再分割できる
    hh_train_datasets = load_dataset(
        "madao33/new-title-chinese", split=["train[:50%]", "train[50%:]"]
    )
    hh_train_datasets
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Datasetsの中身の確認
    #### dict型で帰ってくる
    """)
    return


@app.cell
def _(datasets):
    # 訓練データセットの1個目
    datasets["train"][0]
    return


@app.cell
def _(datasets):
    # 訓練データセットの2個目まで
    datasets["train"][:2]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### データのフィールドを絞って見る時
    #### 上記の出力結果からデータのfeatruesはtitleとcontentがある
    """)
    return


@app.cell
def _(datasets):
    # column_namesフィールドでデータセットにどのようなfeaturesがあるかわかる
    datasets["train"].column_names
    return


@app.cell
def _(datasets):
    # featuresフィールドはfeaturesのデータ型までわかる
    datasets["train"].features
    return


@app.cell
def _(datasets):
    # titleだけを取り出す
    datasets["train"]["title"][:5]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Datasetsによってはテストデータを事前に切り分けていない場合おあるため、
    ### 自分で切り分ける必要がある
    """)
    return


@app.cell
def _(boolq_datasets):
    # Datasetsでsklearnと似たようなメソッドを用意している
    # test_sizeはテストデータの割合を指定、stratify_by_columnはデータラベル不均衡を防ぐためにlabelを指定する
    boolq_datasets["train"].train_test_split(
        test_size=0.1, stratify_by_column="label"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### データの選択とフィルタリング
    """)
    return


@app.cell
def _(datasets):
    # スライス構文で取り出す方法と違って、dictではなくDatasetが返される
    datasets["train"].select([0, 1])
    return


@app.cell
def _(datasets):
    datasets["train"].filter(lambda d: "中国" in d["title"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### データの一括処理
    """)
    return


@app.function
def add_prefix(example):
    example["title"] = "Prefix:" + example["title"]
    return example


@app.cell
def _(datasets):
    prefixed_datasets = datasets.map(add_prefix)
    prefixed_datasets["train"]["title"][:10]
    return


@app.cell
def _(AutoTokenizer):

    # 形態素解析を適用する時の処理
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    def preprocess_function(example):
        model_inputs = tokenizer(
            example["content"], max_length=512, truncation=True
        )
        labels = tokenizer(example["title"], max_length=32, truncation=True)

        model_inputs["label"] = labels["input_ids"]

        return model_inputs
    return (preprocess_function,)


@app.cell
def _(datasets, preprocess_function):
    _processed_datasets = datasets.map(preprocess_function, batched=True)
    _processed_datasets
    return


@app.cell
def _(datasets, preprocess_function):
    # titleとcontentは解析後はいらないので消すべき
    processed_datasets = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )
    processed_datasets
    return (processed_datasets,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 保存とロード
    """)
    return


@app.cell
def _(processed_datasets):
    processed_datasets.save_to_disk("./processed_data")
    return


@app.cell
def _(load_from_disk):
    local_datasets = load_from_disk("./processed_data/")
    local_datasets
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 自前で用意したローカルのデータセットをロードする
    """)
    return


@app.cell
def _(load_dataset):
    csv_datasets = load_dataset(
        "csv", data_files="./data/ChnSentiCorp_htl_all.csv"
    )
    csv_datasets
    return


if __name__ == "__main__":
    app.run()
