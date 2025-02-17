# 卒業研究
卒業研究のリポジトリです。

## フォルダ構造
- `/` : 主なコードが入っているフォルダ
- `/archive` : 使わなくなったコードが入っているフォルダ
- `/util` : その他のユーティリティ関数が入っているフォルダ

## pythonスクリプトについて
python version : 3.10
ライブラリのインストールは以下のコマンドで行ってください。
```bash
pip install -r requirements.txt
```
separator.pyに解析したい音楽ファイルのパスを指定して実行すると、その音楽ファイルのセクションを推定して保存します。
保存フォルダは`/output`です。
```bash
python separator.py /path/to/music/file
```

## ライセンス
[MIT License](./LICENSE)

## 使用したライブラリのライセンス
[Librosa](https://github.com/librosa/librosa/blob/main/LICENSE.md) : ISC License
