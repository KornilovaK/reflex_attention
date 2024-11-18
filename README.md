# reflex_attention
Based on nanoGPT, dataset - openwebtext

```pip install torch numpy transformers datasets tiktoken wandb tqdm inspect```

```I used GPU 1xA100 40GB, RAM 128GB```

```Requires version of pytorch >= 2.0 as there's a flash attention```

## Motivation
- due to a calculation error with model fleets, it is difficult to fit information about a large context in one hidden ([FACTS](http://arxiv.org/pdf/2406.04267 ))
- Decoder layers are trained to predict the following tokens, so the model forgets information from the context when doing only self-attention

## Idea of Reflex attention
- First of 2 layers do only Self-Attention (later SA) using current hidden weights. Other layers' attention = concat[SA, CA1, CA2], where SA - self attention with cuurent weight, CA2 - cross attention (KV - from last previous layer), CA1 - cross attention (KV - from previous previous layer). I'll call it **2 layer reflex attention**. As I used 8 heads and 6 layers, I tried different combinations of heeds (sa-ca2-ca1):
    1) 4-2-2;
    2) 5-2-1;
    3) 3-3-2;

- I also implemented other version of reflex attention. The idea is to use every of previous head. The scheme:
    1) SA_1 (8)
    2) concat[SA_2 (7), CA_1 (1)]
    3) concat[SA_3 (6), CA_2 (1), CA_1 (1)]
    ...
    6) concat[SA_6 (3), CA_5 (1), CA_4 (1), CA_3 (1), CA_2 (1), CA_1 (1)]


## Experiments
I didn't totally completed any training 'cause, for example, the last model needed to be trained during 2.28 days:
1) -  8 heads, 6 layers, n_embed = 512 (as in original paper)
    - vocab_size = 50304, block_size = 256 (from nanoGPT config for small model)
    - batch_size = 32, bias = False, dropout = 0.0, iters = 5000, min_lr = 1e-5

Original attention VS Reflex attention 