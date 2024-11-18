# reflex_attention
Based on nanoGPT, dataset - openwebtext

```pip install torch numpy transformers datasets tiktoken wandb tqdm inspect```

```I used GPU 1xA100 40GB, RAM 128GB```

```Requires version of pytorch >= 2.0 as there's a flash attention```

## Motivation
- due to a calculation error with model fleets, it is difficult to fit information about a large context in one hidden ([FACTS](http://arxiv.org/pdf/2406.04267 ))
- Decoder layers are trained to predict the following tokens, so the model forgets information from the context when doing only self-attention

## Idea of Reflex attention
- First of 2 layers do only Self-Attention (later SA) using current hidden weights. Other layers' attention = concat[SA, CA1, CA2], where SA - self attention with cuurent weight, CA2 - cross attention (KV - from last previous layer), CA1 - cross attention (KV - from previous previous layer). I'll call it **2 layer reflex attention**. As I used 8 heads and 6 layers, I tried different combinations of head numbers (sa-ca2-ca1) and meanung of numbers (maybe we should pay more attention to the nearest layers):
    1) 4-2-2;
    2) 5-2-1;

- I also implemented other version of reflex attention. The idea is to use every of previous head. I'll call it **all layer reflex attention** The scheme:
    1) SA_1 (8)
    2) concat[SA_2 (7), CA_1 (1)]
    3) concat[SA_3 (6), CA_2 (1), CA_1 (1)]
    ...
    6) concat[SA_6 (3), CA_5 (1), CA_4 (1), CA_3 (1), CA_2 (1), CA_1 (1)]


## Experiments
8 heads, 6 layers, vocab_size = 50304, batch_size = 32, bias = False (a little difference with bias=True), min_lr = 1e-5, iters = 5000 (only last models were trained with 7250 iters)

I didn't totally completed any training 'cause, for example, the last model needed to be trained during 2.28 days:
1) n_embed = 512 (as in original paper), block_size = 256 (from nanoGPT config for small model), dropout = 0.0. Original VS 4-2-2 Reflex attention 
<img src="charts/train-val orig-reflex 256-512.jpg" /> <img src="charts/mfu orig-rafelx 256-512.png" />

2) n_embed = 512, block_size = 1024 (from nanoGPT GPT2 training), dropout = 0.0. 4-2-2 Reflex attention 256 VS 1024  block_size(with and without bias=True)
<img src="charts/train-val reflex 256-1024 512.jpg" /> <img src="charts/mfu reflex 256-1024 512.png" />

3) n_embed = 768, block_size = 1024, dropout = 0.0 . 4-2-2 Reflex attention 512 VS 768 n_embed
<img src="charts/train-val reflex 1024 512-768.png" /> <img src="charts/mfu reflex 1024 512-768.png" />

4) n_embed = 768, block_size = 1024, dropout = 0.0. 4-2-2 Reflex 4-2-2 VS 5-2-1 VS all-layer VS original attention
<img src="charts/train-val orig reflex 2-all layer.png" /> <img src="charts/mfu orig reflex 2-all layer.png" /> 
Trained during 7250 models:
    * [Original attention](https://huggingface.co/Eka-Korn/original_gpt) 
    * [5-2-1 Reflex attention](https://huggingface.co/Eka-Korn/reflex-attention_5-2-1_GPT)
    * [all layer Reflex attention](https://huggingface.co/Eka-Korn/reflex-attention_all-layers_GPT)

5) n_embed = 768, block_size = 1024, dropout = 0.1 (there's no reason to use it as we can't overfit here)
<img src="charts/val with dropout.jpg" /> 

6) In [nanoGPT](https://github.com/karpathy/nanoGPT.git) Andrej Karpathy uses 1 Linear at the same time for QKV and then split them for equal parts-heads. I checked what if to use 3 different Linears separately -> the training was muuuuch longer and less effective <img src="charts/1 - 3 linears.png" /> 

### Conclusion
In every experiment Reflex attention showed better results than original even after not so many iters (and I guess if I'd train it more, the difference would be bigger). In addition, it turned out that allocating a large number of heads for the previous layer than for the previous previous one is really better