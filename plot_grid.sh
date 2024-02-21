#!/bin/bash

for size in '7b' '13b' '70b'
do
    for input in 'zh' 'fr' 'de' 'ru' 'en' 
    do 
        for output in 'zh' 'fr' 'de' 'ru' 'en'
        do 
            echo "size: $size, input: $input, output: $output"
            papermill Translation.ipynb ./visuals/executed_notebooks/Translation_Final_${size}_${input}_${output}.ipynb -p model_size $size -p target_lang $output -p input_lang $input
        done 
    done 
done 

for size in '7b' '13b' '70b'
do
    for output in 'zh' 'fr' 'de' 'ru' 'en'
    do 
        echo "size: $size, output: $output"
        papermill Cloze.ipynb ./visuals/executed_notebooks/Cloze_Final_${size}_${output}.ipynb -p model_size $size -p target_lang $output -p input_lang $output
    done
done
