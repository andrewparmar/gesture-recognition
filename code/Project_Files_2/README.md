# CV_Spring2020 Final Project Submission:
# Author: Andrew Parmar
# GID: 

## Package Description
This package is the final project deliverable for the Action Recognition topic of the
CS6476 course. The package contains the final code pacakge containing all source files 
used to generate images and output shows in the final report.


### Description of files

    experiments.py - Execution entry point, all output was generated using this module. 
    core.py - Contains the core classes and functionality of the investigation.
    utils.py - Helper methods
    config.py - Defines constants and settings.
    create_action_compilation_video.py - Script to create a video of mixed actions.
    

### Requirements and Installation

Using this package requires additional files that are not included here due to file size restrictions.
These additional files can be downloaded from [link]

Once downloaded unzip the folder here. The final folder structure should be similar to 

    package_folder
        ├── README.md
        ├── config.py
        ├── core.py
        ├── create_action_compilation_video.py
        ├── cv_proj.yml
        ├── experiment.py
        ├── input_files
        └── utils.py
 

### Video Link

The final output video can be viewed here [link]


## Execution of experiments

    # Activate the virutal enviroment
    conda activate cv_proj
    
    # View help on experiments.py for experiment descriptions and runtimes.
    python experiment.py -h
    
    # Run experiment to generate video. 
    python experiment.py --exp 5
    
    # Run other expriments e.g.
    python experiment.py --exp 2
    python experiment.py --exp 3
    python experiment.py --exp 4
    
Note: --exp 1 generates raw data and trains the classifier, this takes a long time to run.
