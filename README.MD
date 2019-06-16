## Deep Learning of Graph Matching

This is an implementation of the paper "Deep Learning of Graph Matching", CVPR, 2018. 
**Note: This is not an official implementation.**

To run the above code, install the dependecies as listed in `environment.yml`. The easiest way to do this is to setup a conda environemnt

`conda env create -f environment.yml`
`conda activate dlgm`

Then update the parameters in the `config_sintel.py` script according to your hardware availability and needs. 

### Training the network

`python main.py --dataset sintel --data_path path/to/sintel/root`

Note that only sintel dataset is supported as of now. We are not able to exactly reproduce the results of the paper. But the graph matching layers corresponding to optical flow has been tested for correctness. A primitive implementation of the Bistochastic layer is available but is not tested. 

### Contributing
There is lot of scope for improvements with respect to the performance. PR's and discussion in this regard or any general matter on this topic is welcome :)




