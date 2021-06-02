# Overview

<hr/>

# Requirements

## Developers

### Creating conda environment

A conda environment can be created by executing any of the following commands:

<ul>
  <li>
    <div class="snippet-clipboard-content position-relative" data-snippet-clipboard-copy-content=" conda create --name PRTR --file requirements.txt">
        <code>conda create --name PRTR --file requirements.txt</code>
    </div>
  </li>
  <li>
    <div class="snippet-clipboard-content position-relative" data-snippet-clipboard-copy-content=" conda PRTR create -f environment.yml">
      <pre>
        <code>conda PRTR create -f environment.yml</code>
      </pre>
    </div>
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
