# MMSM (Multiscale Markov State Models)

## Author: 
Kessem Clein (Raveh lab)

## Date last updated:
Oct 1st, 2021 or later

## Synopsis: 
Multiscale Markov-State Models (MMSMs) and MMSM-Explore code

## Prerequisties: 
* scipy
* numpy
* matplotlib
* sklearn

## Folder structure:
### base
Abstract base classes defining the interfaces of the different modules

### discretizers
Classes that inherit from BaseDiscretizer. These provide methods for discretizing continuous spaces into discrete sets of states.

### estimators
This module has two submodules for two different types of estimators used by an HMSM.
#### metastable_partition
Classes that inherit from MetastablePartition. These provide methods for estimating partitions of the states of a Markov model, such that the resulting partition is a coarse grained estimate of the Markov model.
#### transitions
Estimators of the transition probabilities between states, based on observed transitions.

### models
#### hmsm
Implementation of the HMSM model, that makes use of the different estimators and optimizers.
#### parallel_hmsm
Implementation of the HMSM model that makes use of parallelization. (Not implemented)

### optimizers
Methods for choosing vertices from an HMSM for adaptive sampling.

### plots
Various methods for creating plots.

### samplers
Classes that inherit from BaseSampler.

### util
Various utility functions used in the package.

## Usage:
TBD

