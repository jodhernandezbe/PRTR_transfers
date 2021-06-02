# Overview

# Requirements

## Ovoiding ModuleNotFoundError and ImportError

If you are working as a Python developer, you should avoid both ```ModuleNotFoundError``` and ```ImportError``` (check the following [link](https://towardsdatascience.com/how-to-fix-modulenotfounderror-and-importerror-248ce5b69b1c)). Thus:

<ol>
  <l>
    Run the following command in order to obtain the RTR_transfers location and then saving its path into the variable PACKAGE
    
    ```
    PACKAGE=$(locate -br '^PRTR_transfers$')
    ```
  </l>
</ol>

If you like, you could check the PACKAGE value by running the following command:

```
echo "$PACKAGE"
```
Finally, run the following 
```
export PYTHONPATH="${PYTHONPATH}:$PACKAGE"
```
