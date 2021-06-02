<hr/>

# Overview

## Project tree

The following is the project tree considering only its most important files for a developer.

```bash

PRTR_transfers
├── ancillary
├── application
├── data_driven
└── data_engineering
    ├── main.py
    ├── extract
    │   ├── config.yaml
    │   ├── main.py
    │   ├── npi_scraper.py
    │   ├── npri_scraper.py
    │   ├── tri_scraper.py
    │   └── output
    ├── load
    │   ├── main.py
    │   ├── industry_sector.py
    │   ├── base.py
    │   └── output
    └── transform
        ├── main.py
        ├── industry_sector_standardizing.py
        ├── npi_transformer.py
        ├── npri_transformer.py
        ├── tri_transformer.py
        └── output

```

<hr/>

# Requirements

## Developers

### Creating conda environment

A conda environment can be created by executing any of the following commands:

<ul>
  <li>
    
     conda create --name PRTR --file requirements.txt
  </li>
  <li>
    
    conda PRTR create -f environment.yml
  </li>
</ul>

The above commands are written assuming that you are in the folder containing the .txt and .yml files, i.e. the root folder PRTR_transfers. 

### Ovoiding ModuleNotFoundError and ImportError

If you are working as a Python developer, you should avoid both ```ModuleNotFoundError``` and ```ImportError``` (see the following [link](https://towardsdatascience.com/how-to-fix-modulenotfounderror-and-importerror-248ce5b69b1c)). Thus, follow the steps below to solve the above mentioned problems:

<ol>
  <li>
    Run the following command in order to obtain the PRTR_transfers project location and then saving its path into the variable PACKAGE
    
    PACKAGE=$(locate -br '^PRTR_transfers$')
  </li>
  <li>
    Check the PACKAGE value by running the following command
    
    echo "$PACKAGE"
   </li>
   <li>
    Run the following command to add the PRTR_transfers project to the system paths
     
    export PYTHONPATH="${PYTHONPATH}:$PACKAGE"
   </li>
</ol>

If you prefer to save the path to the PRTR_transfers project folder as a permanent environment variable, follow these steps:

<ol>
   <li>
    Open the .bashrc file with the text editor of your preference (e.g., Visual Studio Code)
        
    code ~/.bashrc
   </li>
   <li>
    Scroll to the bottom of the file and add the following lines
       
    export PACKAGE=$(locate -br '^PRTR_transfers$')
    export PYTHONPATH="${PYTHONPATH}:$PACKAGE"
   </li>
</ol>

<hr/>
