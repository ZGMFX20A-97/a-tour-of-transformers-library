import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


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
    import pandas as pd
    return AutoModelForSequenceClassification, AutoTokenizer, pd


@app.cell
def _(mo):
    mo.md(r"""
    ### Datasetの作成
    """)
    return


@app.cell
def _(pd):
    from torch.utils.data import Dataset

    class MyDataset(Dataset):

        def __init__(self):
            super().__init__()
            self.data = pd.read_csv("./data/ChnSentiCorp_htl_all.csv")
            self.data = self.data.dropna()

        def __getitem__(self,index):
            return self.data.iloc[index]["review"],self.data.iloc[index]["label"]

        def __len__(self):
            return len(self.data)
        
    return (MyDataset,)


@app.cell
def _(MyDataset):
    dataset = MyDataset()

    for i in range(5):
        print(dataset[i])
    return (dataset,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Datasetの切り分け
    """)
    return


@app.cell
def _(dataset):
    from torch.utils.data import random_split

    trainset,validset = random_split(dataset,lengths=[0.9,0.1])
    len(trainset),len(validset)
    return trainset, validset


@app.cell
def _(mo):
    mo.md(r"""
    ### DataLoaderの作成
    """)
    return


@app.cell
def _(AutoTokenizer):
    import torch
    tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

    def collate_function(batch):
        texts,labels = [],[]
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(texts,max_length=128,padding="max_length",truncation=True,return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs
    return collate_function, tokenizer, torch


@app.cell
def _(collate_function, trainset, validset):
    from torch.utils.data import DataLoader

    train_loader = DataLoader(trainset,batch_size=32,shuffle=True,collate_fn=collate_function)
    valid_loader = DataLoader(validset,batch_size=64,shuffle=False,collate_fn=collate_function)
    return train_loader, valid_loader


@app.cell
def _(train_loader):
    next(enumerate(train_loader))
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
def _(
    model,
    mps_device,
    optimizer,
    torch,
    train_loader,
    valid_loader,
    validset,
):
    def evaluate():
        model.eval()
        acc_num = 0
        with torch.inference_mode():
            for batch in valid_loader:
                batch = {k: v.to(mps_device) for k,v in batch.items()}
                output = model(**batch)
                pred = torch.argmax(output.logits,dim=-1)
                acc_num += (pred.long() == batch["labels"].long()).float().sum()
        return acc_num / len(validset)

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
        
            acc = evaluate()
            print(f"epoch: {epoch}, acc: {acc}")
        
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
