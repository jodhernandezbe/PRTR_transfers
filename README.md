# Overview

# Requirements

## Ovoiding ModuleNotFoundError and ImportError

If you are working as a Python developer, you should avoid both ```ModuleNotFoundError``` and ```ImportError``` (see the following [link](https://towardsdatascience.com/how-to-fix-modulenotfounderror-and-importerror-248ce5b69b1c)). Thus:

<ol>
  <li>
    Run the following command in order to obtain the PRTR_transfers project location and then saving its path into the variable PACKAGE
    
    PACKAGE=$(locate -br '^PRTR_transfers$')
  </li>
  <li>
    Check the PACKAGE value by running the following command
    
    ```
    echo "$PACKAGE"
    ```
   </li>
   <li>
     Run the following command to add the PRTR_transfers project to the system paths
     
     ```
     export PYTHONPATH="${PYTHONPATH}:$PACKAGE"
     ```
   </li>
</ol>
