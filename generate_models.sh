
intermediate_partitions="1 2 4 8"
fc_out_partition="1 2 4 8"
embedding_partitions="1 2 4"
bools="False True"

for emb in $embedding_partitions; do
    for use_conv in $bools; do
        while IFS= read -r model_name
        do
            echo "$model_name $1 1 $emb $use_conv"
            python3 generate_model_batch.py $model_name $1 1 $emb $use_conv
        done < model_names.txt
    done
done