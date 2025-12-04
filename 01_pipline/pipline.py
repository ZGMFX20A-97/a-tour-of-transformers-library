import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    from transformers.pipelines import SUPPORTED_TASKS
    return (SUPPORTED_TASKS,)


@app.cell
def _(mo):
    mo.md(r"""
    ### どんなタスクができるのかを確認する
    """)
    return


@app.cell
def _(SUPPORTED_TASKS):
    for k,v in SUPPORTED_TASKS.items():
        print(k,v["type"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### piepelineの作成と使い方

    タスクの種類でpipelineを作成する、デフォルトは英語のモデル(DISTIL-BERT)

    モデルを指定せずに作るのはあまり良くない
    """)
    return


@app.cell
def _():
    from transformers import pipeline

    pipe1 = pipeline('text-classification')
    return pipe1, pipeline


@app.cell
def _(pipe1):
    pipe1("very good!")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### タスクの種類とモデルを指定して作成する

    中国語のモデルを使ってみる
    """)
    return


@app.cell
def _(pipeline):
    pipe2 = pipeline('text-classification',model='uer/roberta-base-finetuned-dianping-chinese')
    return (pipe2,)


@app.cell
def _(pipe2):
    pipe2('我觉得一般，但也有一些可圈可点之处')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 予めモデルをロードして、pipelineを作成する
    """)
    return


@app.cell
def _(pipeline):
    from transformers import AutoModelForSequenceClassification,AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')
    tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')
    pipe3 = pipeline('text-classification',model=model,tokenizer=tokenizer)
    return (pipe3,)


@app.cell
def _(pipe3):
    pipe3('我觉得不太行')
    return


@app.cell
def _(pipe3):
    # modelがどのデバイスにロードされているのかを確認する(デフォルトではmps->macのGPU)
    pipe3.model.device
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
