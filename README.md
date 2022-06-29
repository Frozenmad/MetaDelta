# MetaDelta++: Improve Generalization of Few-shot System Through Multi-Scale Pretrained Models and Improved Training Strategies

__News!__  
- Serve as a baseline adopted for the Cross-Domain MetaDL competition!
- We've removed all the multiprocessing and uncessary parts to make the logic nice and clean. For the full version, you can refer to the branch neurips2021 or aaai2021.

## Requirements

Please follow the [official website](https://github.com/DustinCarrion/cd-metadl/tree/8c6128120ab8aac331c958b2965d42747d9dbdeb) to set-up environments.  

## Run the code under competition setting

Please follow the [official website](https://github.com/DustinCarrion/cd-metadl/tree/8c6128120ab8aac331c958b2965d42747d9dbdeb) to run the codes.  
For example, cd to the folder of the cd-metadl, and run the following command:
```
cd path/to/cd-metadl
python -m cdmetadl.run --input_data_dir=public_data --submission_dir=path/to/this/folder --output_dir_ingestion=ingestion_output --output_dir_scoring=scoring_output --verbose=False --overwrite_previous_results=True --test_tasks_per_dataset=10
```

Remember to replace the `path/to/cd-metadl` and `path/to/this/folder` to your settings.

__@AutoMLGroup/Meta_Learners__
