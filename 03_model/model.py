import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    from transformers import AutoConfig,AutoModel,AutoTokenizer
    return AutoModel, AutoTokenizer


@app.cell
def _(mo):
    mo.md(r"""
    ### モデルをロードする
    """)
    return


@app.cell
def _(AutoModel):
    model_name = "FacebookAI/xlm-roberta-base"
    model = AutoModel.from_pretrained(model_name)
    model
    return model, model_name


@app.cell
def _(model):
    model.config
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### モデルを使ってみる
    """)
    return


@app.cell
def _(AutoTokenizer, model_name):
    sentence = "私たちは少し先しか見えないかもしれない、しかしそこにすべきことがたくさんあることは確かだ。"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    result1 = tokenizer(sentence,return_tensors="pt")
    result1
    return (result1,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Model Headなしで呼び出す
    """)
    return


@app.cell
def _(model, result1):
    # -baseで終わるモデルはバックボーンモデル -> 自分で下流のタスクについてfine-tunningする必要がある
    output = model(**result1)
    output
    return (output,)


@app.cell
def _(output):
    # Headがないというのは、モデルはただ入力データをエンコードしてそのベクトルを出力するだけ(last_hidden_state)
    output.last_hidden_state.size()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Model Head付きで呼び出す
    """)
    return


@app.cell
def _():
    # テキスト分類のタスクヘッドがついているモデルを呼び出す
    from transformers import AutoModelForSequenceClassification
    return (AutoModelForSequenceClassification,)


@app.cell
def _(AutoModelForSequenceClassification, result1):
    clz_model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
    clz_model(**result1)
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
