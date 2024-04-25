This is a project about neutrino shower direction reconstruction.

**Dataset Format**

The dataset is formatted as following: 
(the training features used are preprocessed, the rows are each unique DOM sorted by first time hit, the columns are x0, y0, z0, time, charge)
````
ve_10_100TeV/
├── newbatch1
│   └── event_0.feather
...
│   └── event_999.feather
│   └── mc_events.json
├── newbatch2
├── newbatch3
...
├── newbatch1000
````

**Dataset Format**

To train the model from scratch, first create a folder named trained_model and an empty file named history.csv in the created folder, then run Train.py.

**Acknowledgments**

Some parts of the codes are adapted from:

* [ChenLi](https://github.com/ChenLi2049/ISeeCube)
* [DrHB](https://github.com/DrHB/icecube-2nd-place/)

who both worked on the IceCube Neutrino Kaggle project data.
