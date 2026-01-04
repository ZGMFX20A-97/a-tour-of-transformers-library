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
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from datasets import load_dataset
    return AutoModelForSequenceClassification, AutoTokenizer, load_dataset


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
    ### DataLoaderの作成
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
    return tokenized_datasets, tokenizer, torch


@app.cell
def _(tokenized_datasets, tokenizer):
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    trainset,validset = tokenized_datasets["train"],tokenized_datasets["test"],
    train_loader = DataLoader(trainset,batch_size=32,shuffle=True,collate_fn=DataCollatorWithPadding(tokenizer))
    valid_loader = DataLoader(validset,batch_size=64,shuffle=False,collate_fn=DataCollatorWithPadding(tokenizer))
    return train_loader, valid_loader


@app.cell
def _(train_loader):
    next(enumerate(train_loader))[1]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### モデルとオプティマイザーの作成
    """)
    return


@app.cell
def _(torch):
    torch.backends.mps.is_available()
    return


@app.cell
def _(AutoModelForSequenceClassification, torch):
    from torch.optim import Adam

    model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
    optimizer = Adam(model.parameters(),lr=2e-5)

    mps_device = torch.device("mps")
    model = model.to(mps_device)
    return model, mps_device, optimizer


@app.cell
def _(mo):
    mo.md(r"""
    ### 訓練と検証
    """)
    return


@app.cell
def _():
    import evaluate

    clf_metrics = evaluate.combine(["accuracy","f1"])
    return (clf_metrics,)


@app.cell
def _(
    clf_metrics,
    model,
    mps_device,
    optimizer,
    torch,
    train_loader,
    valid_loader,
):
    def evaluate():
        model.eval()
        #acc_num = 0
        with torch.inference_mode():
            for batch in valid_loader:
                batch = {k: v.to(mps_device) for k,v in batch.items()}
                output = model(**batch)
                pred = torch.argmax(output.logits,dim=-1)
                clf_metrics.add_batch(predictions=pred.long(),references=batch["labels"].long())
                #acc_num += (pred.long() == batch["labels"].long()).float().sum()
        return clf_metrics.compute()

    def train(epochs=3,log_step=100):
        global_step=0
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(mps_device) for k,v in batch.items()}
                optimizer.zero_grad()
                output = model(**batch)
                output.loss.backward()
                optimizer.step()
                if global_step % log_step == 0:
                    print(f"epoch: {epoch},global_step: {global_step},loss: {output.loss.item()}")
                global_step += 1

            clf = evaluate()
            print(f"epoch: {epoch}, {clf}")
    return (train,)


@app.cell
def _(train):
    train()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 推論
    """)
    return


@app.cell
def _(model, mps_device, tokenizer, torch):
    sen = "我觉得这家酒店不错，饭很好吃！"
    id2label = {0: "ネガティブ", 1: "ポジティブ"}
    model.eval()
    with torch.inference_mode():
        inputs = tokenizer(sen,return_tensors="pt")
        inputs = {k: v.to(mps_device) for k,v in inputs.items()}
        logits = model(**inputs).logits
        pred = torch.argmax(logits,dim=-1)
        print(f"input: {sen} 推論の結果: {id2label.get(pred.item())}")
    return id2label, sen


@app.cell
def _(mo):
    mo.md(r"""
    ### pipelineを使ってみる
    """)
    return


@app.cell
def _(id2label, model, sen, tokenizer):
    from transformers import pipeline

    model.config.id2label = id2label
    pipe = pipeline("text-classification",model=model,tokenizer=tokenizer,device=0)

    pipe(sen)
    return


if __name__ == "__main__":
    app.run()
