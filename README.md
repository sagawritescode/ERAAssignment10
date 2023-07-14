# ERA Assignment 10

Objective: Write a model mentioned in the assignment to be trained on CIFAR10 targeting 90+% accuracy

## Guide to the files:

#### ERA_V1_Assignment10.ipynb

This contains the snapshot of the python notebook which was used for the assignment. Imports the current repo and uses functions and classes from the python files 

#### model.py

Model can be found in `CustomResNet` class 
<img width="602" alt="Screenshot 2023-07-14 at 10 18 48 PM" src="https://github.com/sagawritescode/ERAAssignment10/assets/45040561/e7116683-246d-4227-a265-55346365aecc">

#### train.py
Contains train and test functions 

#### dataloader.py
Downloads CIFAR10 dataset 

#### transforms.py 
Contains image augmentation code. We are using Normalise -> Padding And RandomCrop -> HorizontalFlip -> CoarseDropout

#### utils.py
Contains utils function to print_summary, get_device and get_lr


## Results:

### Analysis:

Running LRFinder multiple times was required to achieve 90+ accuracy. In one run I experimented with an LR near the recommended LR and gave 90+ accuracy  

LR finder graph: </br>
<img width="683" alt="Screenshot 2023-07-14 at 11 21 56 AM" src="https://github.com/sagawritescode/ERAAssignment10/assets/45040561/878aff9d-da54-44b6-859c-0cd73ae4d498">



Testing logs:

```EPOCH: 0
Epoch=0 Loss=355.73187255859375 LR=0.011212184049079756 Batch_id=97 Accuracy=48.61: 100%|██████████| 98/98 [00:24<00:00,  3.94it/s]

Test set: Average loss: 1.5385, Accuracy: 5056/10000 (50.56%)

EPOCH: 1
Epoch=1 Loss=310.3156433105469 LR=0.02188636809815951 Batch_id=97 Accuracy=64.33: 100%|██████████| 98/98 [00:25<00:00,  3.88it/s]

Test set: Average loss: 1.5449, Accuracy: 5736/10000 (57.36%)

EPOCH: 2
Epoch=2 Loss=278.63397216796875 LR=0.03256055214723926 Batch_id=97 Accuracy=71.38: 100%|██████████| 98/98 [00:25<00:00,  3.85it/s]

Test set: Average loss: 0.7565, Accuracy: 7428/10000 (74.28%)

EPOCH: 3
Epoch=3 Loss=260.0733947753906 LR=0.04323473619631902 Batch_id=97 Accuracy=77.69: 100%|██████████| 98/98 [00:25<00:00,  3.83it/s]

Test set: Average loss: 1.0001, Accuracy: 7140/10000 (71.40%)

EPOCH: 4
Epoch=4 Loss=172.15667724609375 LR=0.053771109226638025 Batch_id=97 Accuracy=79.95: 100%|██████████| 98/98 [00:25<00:00,  3.86it/s]

Test set: Average loss: 0.7291, Accuracy: 7484/10000 (74.84%)

EPOCH: 5
Epoch=5 Loss=134.2232208251953 LR=0.05093981343716434 Batch_id=97 Accuracy=81.96: 100%|██████████| 98/98 [00:25<00:00,  3.86it/s]

Test set: Average loss: 0.7468, Accuracy: 7616/10000 (76.16%)

EPOCH: 6
Epoch=6 Loss=170.9170379638672 LR=0.048108517647690655 Batch_id=97 Accuracy=84.39: 100%|██████████| 98/98 [00:25<00:00,  3.81it/s]

Test set: Average loss: 0.5156, Accuracy: 8305/10000 (83.05%)

EPOCH: 7
Epoch=7 Loss=137.58497619628906 LR=0.04527722185821697 Batch_id=97 Accuracy=85.93: 100%|██████████| 98/98 [00:25<00:00,  3.85it/s]

Test set: Average loss: 0.5186, Accuracy: 8367/10000 (83.67%)

EPOCH: 8
Epoch=8 Loss=123.07496643066406 LR=0.042445926068743284 Batch_id=97 Accuracy=87.23: 100%|██████████| 98/98 [00:25<00:00,  3.80it/s]

Test set: Average loss: 0.5025, Accuracy: 8407/10000 (84.07%)

EPOCH: 9
Epoch=9 Loss=109.12909698486328 LR=0.039614630279269605 Batch_id=97 Accuracy=88.55: 100%|██████████| 98/98 [00:25<00:00,  3.82it/s]

Test set: Average loss: 0.5743, Accuracy: 8284/10000 (82.84%)

EPOCH: 10
Epoch=10 Loss=120.31126403808594 LR=0.03678333448979591 Batch_id=97 Accuracy=89.08: 100%|██████████| 98/98 [00:25<00:00,  3.85it/s]

Test set: Average loss: 0.5592, Accuracy: 8304/10000 (83.04%)

EPOCH: 11
Epoch=11 Loss=76.49788665771484 LR=0.033952038700322235 Batch_id=97 Accuracy=90.32: 100%|██████████| 98/98 [00:25<00:00,  3.84it/s]

Test set: Average loss: 0.3969, Accuracy: 8748/10000 (87.48%)

EPOCH: 12
Epoch=12 Loss=87.35979461669922 LR=0.03112074291084855 Batch_id=97 Accuracy=91.18: 100%|██████████| 98/98 [00:25<00:00,  3.84it/s]

Test set: Average loss: 0.4720, Accuracy: 8614/10000 (86.14%)

EPOCH: 13
Epoch=13 Loss=72.54073333740234 LR=0.028289447121374864 Batch_id=97 Accuracy=91.69: 100%|██████████| 98/98 [00:25<00:00,  3.84it/s]

Test set: Average loss: 0.3694, Accuracy: 8834/10000 (88.34%)

EPOCH: 14
Epoch=14 Loss=79.99114227294922 LR=0.025458151331901182 Batch_id=97 Accuracy=92.64: 100%|██████████| 98/98 [00:25<00:00,  3.83it/s]

Test set: Average loss: 0.3703, Accuracy: 8876/10000 (88.76%)

EPOCH: 15
Epoch=15 Loss=71.73462677001953 LR=0.0226268555424275 Batch_id=97 Accuracy=93.30: 100%|██████████| 98/98 [00:25<00:00,  3.81it/s]

Test set: Average loss: 0.3324, Accuracy: 8994/10000 (89.94%)

EPOCH: 16
Epoch=16 Loss=51.221500396728516 LR=0.019795559752953815 Batch_id=97 Accuracy=94.21: 100%|██████████| 98/98 [00:25<00:00,  3.83it/s]

Test set: Average loss: 0.3524, Accuracy: 8941/10000 (89.41%)

EPOCH: 17
Epoch=17 Loss=44.17643356323242 LR=0.016964263963480122 Batch_id=97 Accuracy=95.05: 100%|██████████| 98/98 [00:25<00:00,  3.85it/s]

Test set: Average loss: 0.3298, Accuracy: 9069/10000 (90.69%)

EPOCH: 18
Epoch=18 Loss=51.93223571777344 LR=0.014132968174006444 Batch_id=97 Accuracy=95.74: 100%|██████████| 98/98 [00:25<00:00,  3.84it/s]

Test set: Average loss: 0.3176, Accuracy: 9074/10000 (90.74%)

EPOCH: 19
Epoch=19 Loss=44.78883743286133 LR=0.011301672384532759 Batch_id=97 Accuracy=96.07: 100%|██████████| 98/98 [00:25<00:00,  3.86it/s]

Test set: Average loss: 0.3213, Accuracy: 9114/10000 (91.14%)

EPOCH: 20
Epoch=20 Loss=35.19868850708008 LR=0.008470376595059073 Batch_id=97 Accuracy=96.81: 100%|██████████| 98/98 [00:25<00:00,  3.87it/s]

Test set: Average loss: 0.2984, Accuracy: 9195/10000 (91.95%)

EPOCH: 21
Epoch=21 Loss=23.954055786132812 LR=0.005639080805585388 Batch_id=97 Accuracy=97.19: 100%|██████████| 98/98 [00:25<00:00,  3.84it/s]

Test set: Average loss: 0.2859, Accuracy: 9213/10000 (92.13%)

EPOCH: 22
Epoch=22 Loss=25.89458465576172 LR=0.0028077850161117093 Batch_id=97 Accuracy=97.77: 100%|██████████| 98/98 [00:25<00:00,  3.83it/s]

Test set: Average loss: 0.2762, Accuracy: 9249/10000 (92.49%)

EPOCH: 23
Epoch=23 Loss=8.531848907470703 LR=-2.3510773361976045e-05 Batch_id=97 Accuracy=98.19: 100%|██████████| 98/98 [00:25<00:00,  3.86it/s]

Test set: Average loss: 0.2729, Accuracy: 9274/10000 (92.74%)
```
