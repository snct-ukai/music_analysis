#!/bin/bash
dirpath=$1\
files=$(ls "$dirpath")\
wavfiles=$(echo "$files" | grep -E "\.wav$")

for wavfile in $wavfiles; do
    python ./separator.py "$dirpath/$wavfile"
    echo "Separation done for $wavfile"
done