# MetaDelta++: Improve Generalization of Few-shot System Through Multi-Scale Pretrained Models and Improved Training Strategies

__News!__  
- Serve as a baseline adopted for the Cross-Domain MetaDL competition!

## Requirements

Please follow [official websites](https://github.com/DustinCarrion/cd-metadl/tree/8c6128120ab8aac331c958b2965d42747d9dbdeb) to set-up environments.  

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

__@AutoMLGroup/Meta_Learners__