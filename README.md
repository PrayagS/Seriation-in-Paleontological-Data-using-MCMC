# Seriation in Paleontological data using MCMC

## Abstract

> Given a collection of fossil sites with data about the taxa that occur in each site, the task in biochronology is to find good estimates for the ages or ordering of sites. We describe a full probabilistic model for fossil data. The parameters of the model are natural: the ordering of the sites, the origination and extinction times for each taxon, and the probabilities of different types of errors. We show that the posterior distributions of these parameters can be estimated reliably by using Markov chain Monte Carlo techniques. The posterior distributions of the model parameters can be used to answer many different questions about the data, including seriation (finding the best ordering of the sites) and outlier detection. We demonstrate the usefulness of the model and estimation method on synthetic data and on real data on large late Cenozoic mammals. As an example, for the sites with large number of occurrences of common genera, our methods give orderings, whose correlation with geochronologic ages is 0.95.

Source: the paper (more info [here](#acknowledgements))

## Implementation details

The details of MCMC implementation and the sampling procedures for each parameter of the model can be found in our [report](https://github.com/PrayagS/Seriation_in_Paleontological_Data_using_MCMC/blob/master/Docs/Report.pdf) or in the [paper](https://doi.org/10.1371/journal.pcbi.0020006). Our implementation in pure Python ended up being too slow for actual practice and hence, we used [Cython](https://cython.org/) to introduce types in the code and also to directly call the native C/C++ numpy functions. That ended up increasing the performance but for demonstration purposes where time is key, we still suggest using the original implementation written in C by the authors of the paper. The prepackaged binary found in our repo was also compiled from the C source code. For testing purposes, you can build the Cython code. Instructions are given [below](#usage).

Since MCMC is very sensitive to the initialization step, 100 MCMC chains are ran each having its unique starting data point. We use Python's [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) module to run multiple chains (the actual number varies with the system its being ran on) in parallel. Each chain goes through a *burn-in* period of 10000 samples to converge towards the area of high density. After that, each chain saves 1000 samples out of the 10000 samples it iterates over (this is done to overcome the problem of autocorrelation among the adjacent samples).

From the 100 chains, 8 chains are selected whose expected negative log-likelihood is within one sigma of the best chain. Using the data from these chains, the final seriation and various other plots are generated.

## Dataset

The dataset and the original C source code can be found at the authors' [website](http://www.cis.hut.fi/projects/patdis/paleo/). The dataset contains the values of parameters for European late Cenozoic large land mammals. It was derived from the [NOW database](http://www.helsinki.fi/science/now) on 17 July, 2003.

## Results

The detailed results along with their inferences can be found in our [report](https://github.com/PrayagS/Seriation_in_Paleontological_Data_using_MCMC/blob/master/Docs/Report.pdf). All the code used to generate the correlation coefficients and the plots can be found in [script.py](https://github.com/PrayagS/Seriation_in_Paleontological_Data_using_MCMC/blob/master/script.py).

## Usage

- To generate the required plots and the correlation coefficients, just run [script.py](https://github.com/PrayagS/Seriation_in_Paleontological_Data_using_MCMC/blob/master/script.py). You might want to edit [line no. 60](https://github.com/PrayagS/Seriation_in_Paleontological_Data_using_MCMC/blob/master/script.py#L60) to change the number of parallel processes to be equal to the number of threads in your machine.

- To compile the Cython code:
  ```sh
  $ cd Cython_Implementation
  $ python setup.py build_ext --inplace
  ```
  This will take the .cyx files and generate their counterpart C files. Those generated files will be compiled by `gcc` to generate binaries that can be run on your machine.

## Acknowledgements

Paper Source: Seriation in Paleontological Data Using Markov Chain Monte Carlo Methods
Puolamäki K, Fortelius M, Mannila H (2006) Seriation in Paleontological Data Using Markov Chain Monte Carlo Methods. PLOS Computational Biology 2(2): e6. https://doi.org/10.1371/journal.pcbi.0020006
