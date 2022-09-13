# Quantization Aware Training for BERT models

## Private repo do not share outside of lab (for now)

## Getting Started

Build BERT Transformer model from config or using pretrained [Tensorflow models](https://github.com/google-research/bert)
<p align="center">
  <img src="figures/BERT_Architecture.png" width=400 />
</p>

## Example
### Downloading model
```
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip
unzip uncased_L-2_H-128_A-2.zip -d uncased_L-2_H-128_A-2
```
### Loading model into Tensorflow
```
import ConvertModel

model_dir = "uncased_L-2_H-128_A-2"
bert_encoder = ConvertModel.from_tf1_checkpoint(model_dir) # Tensorflow BERT models were trained using TF1
```

### Creating model from config.json
ExampleConfig.json
```
{
    "hidden_size": 128,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "vocab_size": 30522,
    "hidden_dropout_prob": 0.1,
    "num_attention_heads": 2,
    "type_vocab_size": 2,
    "max_position_embeddings": 512,
    "num_hidden_layers": 2,
    "intermediate_size": 512,
    "attention_probs_dropout_prob": 0.1
}
```
### Loading from config.json
```
import ConvertModel
bert_encoder = ConvertModel.from_config("ExampleConfig.json")
```