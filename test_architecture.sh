#!/bin/bash
# 128 256 512 576 
#listVar="64 128 256 512 576 640 704 784 1024 2048"
listVar="3072 4096"
echo "NEW TEST" >> results.txt
for i in $listVar; do
    n_success=0
    n_fail=0
    n_not_generated=0
    n_unknown=0
    for var in {1..5}
    do
        rm tflite_models/arch_test_int8.tflite
        out1=$(python3 architecture_testing.py $i 2>&1)
        out2=$(edgetpu_compiler tflite_models/arch_test_int8.tflite 2>&1)
        if [[ $out2 == *"Compilation failed!"* ]]; then
            let "n_fail=n_fail+1"
        elif [[ $out2 == *"opening file for reading"* ]]; then
            let "n_fail=n_fail+1"
            let "n_not_generated=n_not_generated+1"
            echo "[NOT GENERATED]"
            echo $out1
            let "var=var-1"
        elif [[ $out2 == *"Compilation succeeded!"* ]]; then
            let "n_success=n_success+1"
        else
            let "n_fail=n_fail+1"
            let "n_unknown=n_unknown+1"
        fi
    done
    echo "Summary for $i d_model" >> results.txt
    echo "success: $n_success   fail: $n_fail" >> results.txt
    echo "Summary for $i d_model"
    echo "success: $n_success   fail: $n_fail"
    echo ""
done
