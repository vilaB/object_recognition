S²-LOR: Supervised Stream Learning for Object
===============================================

This is an implementation of the S-LOR system proposed for the IbPRIA-2023 congress.

S²-LOR is an ensemble-based system that allows learning over time based from non-stationary data stream. This system combines learning operations with query/prediction operations. S²-LOR implements feature extraction using a object recognition pretrained model based on ResNet-50, and classification process using Sklearn SVMs. This method outperforms most of the results of CORE-50, while making minimal use of memory storage. It also demonstrates how the system is able to achieve results in the order of resource-intensive solutions.

Formally, S²-LOR, takes a video sequence and passes it through the pre-trained model of the ResNet-50, obtaining its embedding vectors. The system then evaluates those vectors in the ensembles, obtaining a score for each classifier. Once that has been obtained, an overall score is determined and the identity associated with the sample is determined.  In case the prediction does not match the correct identity, a new classifier is introduced to improve the predictions.

![Schema](https://user-images.githubusercontent.com/58976466/236062428-79527429-7d88-4c73-87cd-39f2c5d9442a.png)

## Dependecies

We have tested the code with the following packages and versions:

- Python 3.10.6
- tensorflow 2.10.0
- tensorflow-hub 0.8.0
- pillow 9.4.0
- opencv 4.6.0
- scikit-learn 1.2.1


We recommend setting up a `conda` environment with these same package versions:
```
conda create -n slor_dev python=3.10.6
conda activate slor_dev
conda install tensorflow==2.10.0 tensorflow-hub==0.8.0 pillow==9.4.0 opencv==4.6.0 scikit-learn==1.2.1
```

## Repo Structure & Descriptions
* [feature_extraction.py](./FeatureExtraction/feature_extraction): ResNet50 configuration for generate de embedding vectors
* [process_results](./process_results): folder where the result tables of the desired experiment will be stored.
    * [main.sh](./process_results/main.sh): script for generate the resul tables of the [id_experiment] experiment
* [main.py](./main.py): script to run each experiment, with the proposed configuration
* [SVM.py](./agent/SVM.py): file with the implementation code for each weak SVM classifier
* [ensemble.py](./agent/ensemble.py): ensemble implementation, with de addition mechanisms for training and initialization
* [agent.py](./agent/agent.py): implementation detail for each mode of processing
* [constants.py](./agent/constants.py): configuration details for decision mechanism.
* [tools.py](./agent/tools.py): functions to generate de scores for each frame and sequence and to generate the negatives.
* [load_data.py](./load_data.py): script to load and normalize the CORE50 data.
* [main.sh](./main.sh): script to execute 10 runs for the experiment id indicated by command line.


## Running the experiments
#### Previous configurations
You must download de cropped database(`cropped_128x128_images.zip`) from [CORE50 page](https://vlomonaco.github.io/core50/index.html#download). Then you have to unzip the folder inside the project home directory (`object_recognition`)

#### Feature extraction
In order to make the feature extraction process faster, it is performed offline. To do this, once the `core50_128x128` folder is obtained, the following commands are executed.
```
conda activate slor_dev
cd FeatureExtraction
python3 featureextraction.py
cd ..
```

#### To run S²-LOR proccesing of CORE50 database
```
bash main.sh experiment_name
```
It is recommended to change the name of the experiment depending on what you want to test, trying to give names that are representative of what you want to evaluate. For example, `supervised_no_limit` for a supervised approach with no classifier number limit.

#### Files generated from the streaming experiment
1. A folder with the experiment title is generated under the folder `experiments`.`[num_run]` denotes the repetition about which information is stored. This folder contains:
    - `[num_run]results_sup.txt` which contains the accuracy results for each adaptation steps.
    - `[num_run]ensemble_size_sup.txt` which contains the final ensemble size for each adaptation steps.
    - `output_[num_run]` which shows the stacktrace of each run.

2. In the folder `process_results`, there are a script called `main.sh`. Running this script you could obtain the result tables of the runned experiment. You must execute:
    ```
    cd process_results
    bash main.sh experiment_name
    ```
 
    
# Contact
If you have any questions about this code, please do not hesitate to contact me at gabrivb@outlook.es or gabriel.vilarino@rai.usc.es.
