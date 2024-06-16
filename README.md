# Overview
This project includes the code developed to generate the results of the following paper:
Qing An, Mehdi Zafari, Chris Dick, Santiago Segarra, Ashutosh Sabharwal, Rahman Doost-Mohammady, "ML-Based Feedback-Free Adaptive MCS Selection for Massive Multi-User MIMO", Asilomar 2023 (PDF): https://ieeexplore.ieee.org/document/10476866.

# Getting Started
## Prerequisites
* PyTorch 
* Python 3

## DataSet
 
Dataset is in dataset_process folder. We use post_data.py to process raw_dataset and save post_processed dataset in rdy_dataset.

We use test.py to check inter-user correlation in each dataset. (correlated / uncorrelated scenarios)

Because space limitation, we put our dataset here: https://drive.google.com/drive/folders/14Y2T6d2ctpwNIt9dlIJguMCO-mFgZ4n8?usp=sharing

## Train
Once you have created the dataset, start training ->
```
python main.py (Change training configuratons in opts.py) 

```

## Note 
* All the weights will be saved to the snapshots folder 
* To resume Training from any checkpoint, Use
```
--resume_path <path-to-model> 
```

## Acknowledgement
This project was funded in part by the U.S. National Science Foundation under Grant CNS-2016727.

## Contact
Qing An (qa4@rice.edu)

