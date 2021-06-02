# Overview

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
    
    <code>PACKAGE=$(locate -br '^PRTR_transfers$')</code>
  </li>
  <li>
    Check the PACKAGE value by running the following command
    
    <code>echo "$PACKAGE"</code>
   </li>
   <li>
    Run the following command to add the PRTR_transfers project to the system paths
     
    <code>export PYTHONPATH="${PYTHONPATH}:$PACKAGE"</code>
   </li>
</ol>

<hr/>
