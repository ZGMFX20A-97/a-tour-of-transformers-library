import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    from transformers import AutoTokenizer
    return (AutoTokenizer,)


@app.cell
def _(mo):
    mo.md(r"""
    ### tokenizerのロードと保存
    """)
    return


@app.cell
def _(AutoTokenizer):
    # モデルの名前を渡して、モデルに対応したtokenizerがロードされる。
    model_name = "tabularisai/multilingual-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer
    return model_name, tokenizer


@app.cell
def _(tokenizer):
    # tokenizerをローカルに保存する
    tokenizer.save_pretrained("./tokenizer")
    return


@app.cell
def _(AutoTokenizer):
    # ローカルからtokenizerをロードする
    tokenizer_local = AutoTokenizer.from_pretrained("./tokenizer/")
    tokenizer_local
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 形態素分析
    """)
    return


@app.cell
def _(tokenizer):
    sentence = "吾輩は猫である"
    tokens = tokenizer.tokenize(sentence)
    tokens
    return sentence, tokens


@app.cell
def _(mo):
    mo.md(r"""
    ### 各文字、単語をインデックスへマッピングした辞書を見てみる
    """)
    return


@app.cell
def _(tokenizer):
    tokenizer.vocab
    return


@app.cell
def _(tokenizer):
    # 辞書のサイズ
    tokenizer.vocab_size
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### tokensの変換
    """)
    return


@app.cell
def _(tokenizer, tokens):
    # tokensをid配列へ変換する
    ids_from_tokens = tokenizer.convert_tokens_to_ids(tokens)
    ids_from_tokens
    return (ids_from_tokens,)


@app.cell
def _(ids_from_tokens, tokenizer):
    # id配列をtokensへ変換する
    tokens_from_ids = tokenizer.convert_ids_to_tokens(ids_from_tokens)
    tokens_from_ids
    return (tokens_from_ids,)


@app.cell
def _(ids_from_tokens, tokens_from_ids):
    map = {t:ids_from_tokens[i] for i,t in enumerate(tokens_from_ids)}
    map
    return


@app.cell
def _(tokenizer, tokens_from_ids):
    # tokensを文字列へ変換する
    str = tokenizer.convert_tokens_to_string(tokens_from_ids)
    str
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### より便利な方法
    """)
    return


@app.cell
def _(sentence, tokenizer):
    # 文字列をid配列へ変換するときエンコードとも呼ばれる
    ids_by_encoding = tokenizer.encode(sentence,add_special_tokens=False)
    ids_by_encoding
    return (ids_by_encoding,)


@app.cell
def _(ids_by_encoding, tokenizer):
    # id配列を文字列へ変換するときデコードとも呼ばれる
    str_by_decoding = tokenizer.decode(ids_by_encoding)
    str_by_decoding
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### paddingとtruncate
    """)
    return


@app.cell
def _(sentence, tokenizer):
    # モデルは固定長のテンソルしか受け付けないため、シーケンスが短い時は充填する必要がある
    ids_with_padding = tokenizer.encode(sentence,padding="max_length",max_length=15)
    ids_with_padding
    return


@app.cell
def _(sentence, tokenizer):
    # モデルは固定長のテンソルしか受け付けないため、シーケンスが長い時は切断する必要がある
    ids_with_truncate = tokenizer.encode(sentence,truncation=True,max_length=3)
    ids_with_truncate
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 直接tokenizerでもエンコードできる
    """)
    return


@app.cell
def _(sentence, tokenizer):
    # attentionマスクで1は実のデータを指している、0はpaddingで充填したデータを指し示している
    # 0は学習不要で、ただ固定長ベクトルのために埋めているだけ)
    inputs = tokenizer(sentence,padding="max_length",max_length=15)
    inputs
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### バッチを処理する場合
    """)
    return


@app.cell
def _(tokenizer):
    # バッチを処理する時は特別な処理が必要でなく、そのまま渡してもOK
    sentences = ["吾輩は猫である","吾輩は犬である","吾輩はハムスターである"]
    result = tokenizer(sentences)
    result
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Fast/Slow Tokenizerの違い
    """)
    return


@app.cell
def _():
    import time

    sample = "私たちは少し先しか見えないかもしれない、しかしそこにすべきことがたくさんあることは確かだ。"

    return sample, time


@app.cell
def _(AutoTokenizer, model_name):
    # Fast TokenizerはRustで実装、処理の速度が速い
    fast_tokenizer = AutoTokenizer.from_pretrained(model_name)
    fast_tokenizer
    return (fast_tokenizer,)


@app.cell
def _(AutoTokenizer, model_name):
    # Slow TokenizerはPythonで実装、処理の速度が遅い
    slow_tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)
    slow_tokenizer
    return (slow_tokenizer,)


@app.cell
def _(fast_tokenizer, sample, time):
    _start_time = time.perf_counter()
    for _i in range(10000):
        fast_tokenizer(sample)
    _end_time = time.perf_counter()
    _duration = _end_time - _start_time
    print(f"{_duration}秒")
    return


@app.cell
def _(sample, slow_tokenizer, time):
    _start_time = time.perf_counter()
    for _i in range(10000):
        slow_tokenizer(sample)
    _end_time = time.perf_counter()
    _duration = _end_time - _start_time
    print(f"{_duration}秒")
    return


@app.cell
def _(fast_tokenizer, sample):
    # Fast Tokenizerはreturn_offsets_mappingのオプションが使える
    result2 = fast_tokenizer(sample,return_offsets_mapping=True)
    result2
    return


@app.cell
def _(sample, slow_tokenizer):
    # Slow Tokenizerは使えない(エラーが出る)
    try:
        slow_tokenizer(sample,return_offsets_mapping=True)
    except NotImplementedError:
        print("NotImplementedErrorがraiseされた")
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
