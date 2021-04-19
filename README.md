# MetaDelta: A Meta-Learning System for Few-shot Image Classification

_note: this version is only for publication, we will re-organize and clean our code for better research use later_

## Requirements

Please follow [official websites](https://github.com/ebadrian/metadl/tree/master/starting_kit) to set-up environments. 

Apart from the environments above, we also need following packages:
```
pytorch == 1.6.0
lightgbm
```
You can run the metadl test using this repo directly which will help you handle the environments above automatically.

## Run the code under competition setting

First, you need to activate your environments:
```
$ conda activate metadl
```

Anywhere outside this folder, make a new folder and cd to it. Then you can run the official code:
```
$ mkdir paths/to/your/runtime/folder
$ cd paths/to/your/runtime/folder
$ python -m metadl.core.run --meta_dataset_dir=abs/path/to/dataset --code_dir=abs/path/to/this/folder
```

## Customize to other dataset setting

The running method is the same. You just need to change the class number in model.py line 63.

```python
    'class_number': xxx # change to your total num_class in meta-train dataset
```

__@AutoMLGroup/Meta_Learners__