from spleeter.separator import Separator
import sys

if __name__ == '__main__':
    # インプット音源ファイルを指定
    input_file = "./audio_example.mp3"
    # 分離モードを指定
    separator = Separator("spleeter:2stems")

    # インプットファイルと出力ディレクトリを指定して分離実行
    separator.separate_to_file(input_file, "./output-python")
