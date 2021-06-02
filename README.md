# Overview

```bash

PRTR_transfers
├── ancillary
│   └── NPI_columns_for_using.yaml
├── application
├── data_driven
└── data_engineering
    ├── extract
    │   ├── common.py
    │   ├── config.yaml
    │   ├── main.py
    │   ├── npi_scraper.py
    │   ├── npri_scraper.py
    │   ├── output
    │   ├── __pycache__
    │   │   ├── common.cpython-39.pyc
    │   │   ├── main.cpython-39.pyc
    │   │   ├── npi_scraper.cpython-39.pyc
    │   │   ├── npri_scraper.cpython-39.pyc
    │   │   └── tri_scraper.cpython-39.pyc
    │   └── tri_scraper.py
    ├── load
    │   ├── base.py
    │   ├── industry_sector.py
    │   ├── main.py
    │   ├── output
    │   │   ├── PRTR_transfers_v_MySQL.sql
    │   │   └── PRTR_transfers_v_PostgreSQL.sql
    │   └── __pycache__
    │       ├── base.cpython-39.pyc
    │       └── industry_sector.cpython-39.pyc
    ├── main.py
    └── transform
        ├── common.py
        ├── industry_sector_standardizing.py
        ├── main.py
        ├── npi_transformer.py
        ├── npri_transformer.py
        ├── output
        ├── __pycache__
        │   ├── common.cpython-39.pyc
        │   ├── main.cpython-39.pyc
        │   └── npi_transformer.cpython-39.pyc
        └── tri_transformer.py


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

<hr/>
