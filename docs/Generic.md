# Generic

To make a consistent coding style and model design, the VGGbase platform attempts to follow a generic rule of abbreviations, and variables/sarguments naming, which, as a result, helps the developer and user to work smoothly. 


* [Abbreviations](#Abbreviations)
* [Arguments naming](#arguments)


## Abbreviations

The frequent usage of abbreviations in necessary scenarios is welcomed, especially when abbreviations can significantly reduce code redundancy. Pre-defined abbreviations are listed below:

| Full name            | Abbreviation | Example                        |
|----------------------|--------------|--------------------------------|
| number of _xxx_      | n_xxx        | n_channels,  n_features        |
| input _xxx_          | ipt_xxx      | ipt_samples, ipt_noises        |
| desired _xxx_ as in  | in_xxx       | in_n_channels, in_n_features   |
| desired _xxx_ as out | out_xxx      | out_n_channels, out_n_features |
| attention            | attn         | attn_dropout                   |
| probability          | prob         | attn_prob                      |
| images               | rgbs or imgs | ipt_rgbs, ipt_imgs             |
| token of _xxx_       | txxx         | tq, tvq                        |
| projection           | proj         | proj_x, proj_q                 |
| block                | blk          | n_blk                          |
| embedding            | embed        | embed_n_features               |

## Arguments naming

All arguments of the `forward` function should follow the rule:

1. `x_` should be the prefix for any images or text arguments. For instance, `x_rgb`, `x_text`.
2. Once the argument can vary with the task, its name should follow what this argument expects to 
    do. For instance, in cross-modal transformer, as the argument can be either `text` or `boxes` and is used as 
    the query, its name is set to be `query`.
3. All images should be named to be `rgb`, while all language queries related to the text should be named as `text`.
4. The general purpose of arguments should be called `x_samples`.
