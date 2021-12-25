# MetaDelta++: Improve Generalization of Few-shot System Through Multi-Scale Pretrained Models and Improved Training Strategies

_note: this version is only for publication, we will re-organize and clean our code for better research use later_

## Requirements

Please follow [official websites](https://github.com/ebadrian/metadl/tree/master/starting_kit) to set-up environments. 

Apart from the environments above, you also need to install the requirements listed in requirements.txt.

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