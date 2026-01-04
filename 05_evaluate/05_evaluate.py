import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import evaluate
    return evaluate, mo


@app.cell
def _(mo):
    mo.md(r"""
    ### 評価指標の関数を確認してみる
    """)
    return


@app.cell
def _(evaluate):
    evaluate.list_evaluation_modules()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 評価用の関数をロードする
    """)
    return


@app.cell
def _(evaluate):
    accuracy = evaluate.load("accuracy")
    return (accuracy,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 関数の説明を確認する
    """)
    return


@app.cell
def _(accuracy):
    print(accuracy.description)
    return


@app.cell
def _(accuracy):
    print(accuracy.inputs_description)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 評価指標の計算
    """)
    return


@app.cell
def _(accuracy):
    # イテレブルに計算

    for ref,pred in zip([0,1,0,1],[1,0,0,1]):
        accuracy.add(reference=ref,prediction=pred)
    accuracy.compute()
    return


@app.cell
def _(accuracy):
    # バッチで渡す時はadd_batchを使う

    for refs,preds in zip([[0,1],[0,1]],[[1,0],[0,1]]):
        accuracy.add_batch(references=refs,predictions=preds)
    accuracy.compute()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 複数指標の計算
    """)
    return


@app.cell
def _(evaluate):
    clf_metrics = evaluate.combine(["accuracy","f1","recall","precision"])
    clf_metrics
    return (clf_metrics,)


@app.cell
def _(clf_metrics):
    clf_metrics.compute(predictions=[0,1,0],references=[0,1,1])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 評価結果の比較と可視化
    """)
    return


@app.cell
def _():
    from evaluate.visualization import radar_plot
    return (radar_plot,)


@app.cell
def _():
    data = [
        {"accuracy": 0.99,"precision": 0.8,"f1": 0.95,"latency_in_seconds": 33.6},
        {"accuracy": 0.98,"precision": 0.87,"f1": 0.91,"latency_in_seconds": 11.2},
        {"accuracy": 0.98,"precision": 0.78,"f1": 0.88,"latency_in_seconds": 87.6},
        {"accuracy": 0.88,"precision": 0.78,"f1": 0.81,"latency_in_seconds": 101.6},
    ]
    model_names = ["Model 1","Model 2","Model 3","Model 4"]
    return data, model_names


@app.cell
def _(data, model_names, radar_plot):
    plot = radar_plot(data=data,model_names=model_names)
    plot
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
