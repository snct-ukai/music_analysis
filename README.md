# 音楽解析
このリポジトリは卒業研究のためのコードを記録するために作っています。
ここに含まれるコードはMIT Licenseに則って自由に使用しても構いませんが、使っているライブラリにより強いLicenseがある場合はそちらを優先して使用してください。

# フォルダ構造
- `/` : 主なコードが入っているフォルダ
- `/archive` : 使わなくなったコードが入っているフォルダ
- `/chord` : コード進行解析に関するコードが入っているフォルダ
- `freq_analyze` : 音楽の周波数解析に関するコードが入っているフォルダ
- `/util` : その他のユーティリティ関数が入っているフォルダ

# pythonスクリプトについて
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

# ライセンス
このリポジトリのコードはMIT Licenseに則って公開されています。
[MIT License](./LICENSE)

# 使用したライブラリのライセンス
[Librosa](https://github.com/librosa/librosa/blob/main/LICENSE.md) : ISC License
