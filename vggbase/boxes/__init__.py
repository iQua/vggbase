"""
Given one image and one text query containing `P` phrases, VG aims to semantically 
align each phrase with the corresponding regions in this image. 

Different from the uni-modal detection task, such as object detection, it is generally 
no specific class label assigned to each bboxes as each bboxes solely corresponds to 
one of the `P` phrases. 
For example, 
    - the Flickr30k Entities dataset focusing on phrase localization includes the phrase' 
    category as the label for each bbox. Thus, the box label is actually the phrase label.
    - the ReferItGame and Refcoco-related datasets do not assign meaningful labels to bbox. 

In order to be compatible with other detection tasks and those datasets with different 
structures, VGGbase proposes two concepts:
    - labels
    - bbox_ids
    Please see `bbox_generic.py` to see how they are assigned to bboxes.

To be specific, the `labels` is the phrase category, which should be given by the original dataset. 
For example, 
    as there are eight/8 categories of noun phrases in the Flickr30k Entities dataset, 
    the `labels` exists in the range [0, 7]. These categories are {"animals": 0, "bodyparts": 1, 
    "clothing": 2, "instruments": 3, "notvisual": 4, "other": 5, "people": 6, "scene": 7, 
    "vehicles": 8}.
	See `phrases_classes.json` under the data folder for details.

As no label-related information is given by ReferItGame and Refcoco-related datasets, the 
`labels` will maintain None during the whole VGGbase. 

However, these labels for noun phrases cannot be utilized to identify bboxes because bboxes 
for different noun phrases can be assigned the same label when different noun phrases belong 
to the same category.  
For example
    Still in Flickr30k Entities dataset, for phrases `one arm`, `the chest`, and `her head` 
    in one text query, they all belong to the category `bodyparts`, thus having label `1`.

Under such cases, for bboxes with the same label, no one can figure out which phrases they 
belong to.

Therefore, `bbox_ids` is introduced to fix this issue. 
For one text query, the order of its noun phrases maintains the same throughout the 
learning process. 
This motivates VGGbase to utilize the index of these phrases in the text query as their 
temporary ids to identify different phrases. 

Consequently, the `bbox_ids` is the order of the corresponding noun phrase in the text query. 
For example, when a text query contains `P` phrases, the range of `bbox_ids` will be [0, P-1]. 

Therefore, once bboxes have the same `bbox_ids`, they belong to the responses of one specific 
phrase, which can be accessed by these `bbox_ids`.


"""
