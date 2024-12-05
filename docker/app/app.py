from spleeter.separator import Separator
import sys

if __name__ == '__main__':
    try:
        input_file = sys.argv[1]
    except:
        print("Usage: python app.py <input_file>")
        sys.exit(1)
    # 分離モードを指定

    filename = input_file.split("/")[-1]
    filename = filename.split(".")[0]
    separator = Separator("spleeter:4stems")

    # インプットファイルと出力ディレクトリを指定して分離実行
    separator.separate_to_file(input_file, f"./output-python/{filename}")
