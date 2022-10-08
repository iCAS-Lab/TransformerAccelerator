#!/bin/bash
# 128 256 512 576 
listVar="768 3072 4096"
listVar1="768 3072 4096"
echo "NEW TEST" >> results_multiple_conv.txt
for i in $listVar1; do
    for j in $listVar; do
        n_success=0
        n_fail=0
        n_not_generated=0
        n_unknown=0
        for var in {1..2}
        do
            rm tflite_models/arch_test_int8_2.tflite
            out1=$(python3 architecture_testing2.py $i $j 2>&1)
            out2=$(edgetpu_compiler tflite_models/arch_test_int8_2.tflite 2>&1)
            if [[ $out2 == *"Compilation failed!"* ]]; then
                let "n_fail=n_fail+1"
            elif [[ $out2 == *"opening file for reading"* ]]; then
                let "n_fail=n_fail+1"
                let "n_not_generated=n_not_generated+1"
                let "n_unknown=n_unknown+1"
                echo "[NOT GENERATED]"
                echo $out1
                let "var=var-1"
            elif [[ $out2 == *"Compilation succeeded!"* ]]; then
                let "n_success=n_success+1"
            else
                let "n_fail=n_fail+1"
                let "n_unknown=n_unknown+1"
                echo "Failed for some unknown reason"
                echo $2
            fi
        done
        echo "Summary for $i d_model $j emb_size" >> results_multiple_conv.txt
        echo "success: $n_success   fail: $n_fail" >> results_multiple_conv.txt
        echo "Summary for $i d_model $j emb_size"
        echo "success: $n_success   fail: $n_fail   unk_failures: $n_unknown"
        echo ""
    done
done
