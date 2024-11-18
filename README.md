# reflex_attention
Based on nanoGPT, dataset - openwebtext

```pip install torch numpy transformers datasets tiktoken wandb tqdm inspect```

```I used GPU 1xA100 40GB, RAM 128GB```

```Requires version of pytorch >= 2.0 as there's a flash attention```

Experiments (I didn't totally completed any training 'cause, for example, the last model needed to be trained during 2.28 days):

    1) Firstly, I trained a model with simple decoder architecture:
        -  8 heads, 6 layers, n_embed = 512 (as in original paper)
        - vocab_size = 50304, block_size = 256 (from nanoGPT config for     small model)
        - batch_size = 32, bias = False, dropout = 0.0, iters = 5000, min_lr = 1e-5
    And then customised model with the same configs but with reflex attention blocks 