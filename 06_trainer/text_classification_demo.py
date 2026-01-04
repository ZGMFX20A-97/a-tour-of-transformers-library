import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    ### テキスト分類モデルのデモ
    """)
    return


@app.cell
def _():
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments
    from datasets import load_dataset
    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        load_dataset,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Datasetをロードする
    """)
    return


@app.cell
def _(load_dataset):
    dataset = load_dataset("csv",data_files="./data/ChnSentiCorp_htl_all.csv",split="train")
    dataset = dataset.filter(lambda x:x["review"] is not None)
    dataset
    return (dataset,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Datasetの切り分け
    """)
    return


@app.cell
def _(dataset):
    datasets = dataset.train_test_split(test_size=0.1)
    datasets
    return (datasets,)


@app.cell
def _(mo):
    mo.md(r"""
    ### データの前処理
    """)
    return


@app.cell
def _(AutoTokenizer, datasets):
    import torch
    tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

    def process_function(examples):
        tokenized_examples = tokenizer(examples["review"],max_length=128,truncation=True)
        tokenized_examples["labels"] = examples["label"]
        return tokenized_examples

    tokenized_datasets = datasets.map(process_function,batched=True,remove_columns=datasets["train"].column_names)
    tokenized_datasets
    return tokenized_datasets, tokenizer


@app.cell
def _(mo):
    mo.md(r"""
    ### モデルの作成
    """)
    return


@app.cell
def _(AutoModelForSequenceClassification):
    from torch.optim import Adam

    model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")

    # Trainer内でmpsが使えるかどうかで自動的に判断するため、以下のコードは不要になる
    #optimizer = Adam(model.parameters(),lr=2e-5)

    #mps_device = torch.device("mps")
    #model = model.to(mps_device)
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 評価用関数の作成
    """)
    return


@app.cell
def _():
    import evaluate

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def eval_metric(eval_predict):
        predictions,labels = eval_predict
        predictions = predictions.argmax(axis=-1)
        acc = acc_metric.compute(predictions=predictions,references=labels)
        f1 = f1_metric.compute(predictions=predictions,references=labels)
        acc.update(f1)
        return acc
    return (eval_metric,)


@app.cell
def _(mo):
    mo.md(r"""
    ### TrainingArgumentsの作成
    """)
    return


@app.cell
def _(TrainingArguments):
    train_args = TrainingArguments(output_dir="./checkpoints",
                                   per_device_train_batch_size=64,
                                   per_device_eval_batch_size=128,
                                   logging_steps=10,
                                   eval_strategy="epoch",
                                   learning_rate=2e-5,
                                   weight_decay=0.01,
                                   metric_for_best_model="f1",
                                  )
    train_args
    return (train_args,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Trainerの作成
    """)
    return


@app.cell
def _(Trainer, eval_metric, model, tokenized_datasets, tokenizer, train_args):
    from transformers import DataCollatorWithPadding

    trainer=Trainer(model=model,
                    args=train_args,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["test"],
                    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                    compute_metrics=eval_metric)
    return (trainer,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 訓練と検証
    """)
    return


@app.cell
def _(trainer):
    trainer.train()
    return


@app.cell
def _(trainer):
    trainer.evaluate()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 推論
    """)
    return


@app.cell
def _(tokenized_datasets, trainer):
    trainer.predict(tokenized_datasets["test"])
    # sen = "我觉得这家酒店不错，饭很好吃！"
    # id2label = {0: "ネガティブ", 1: "ポジティブ"}
    # model.eval()
    # with torch.inference_mode():
    #     inputs = tokenizer(sen,return_tensors="pt")
    #     inputs = {k: v.to(mps_device) for k,v in inputs.items()}
    #     logits = model(**inputs).logits
    #     pred = torch.argmax(logits,dim=-1)
    #     print(f"input: {sen} 推論の結果: {id2label.get(pred.item())}")
    return


if __name__ == "__main__":
    app.run()
