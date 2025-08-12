#### Install LLaVA.

```shell
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
```

please note this process will install torch 2.1, which is different from taht in *gppllie* conda env. You can install llava within a new conda env, or you can revise the torch and cudatoolkit version in the *gppllie* env.


#### Evaluations

<summary>Calculate glocal score.</summary>
    
```shell
python eval_scripts/llava_v1.5/eval_llie_global.py
```

<summary>Calculate local prior.</summary>
    
```shell
python eval_scripts/llava_v1.5/eval_llie_local.py
```


This pipeline is built upon the [Q-Instruct](https://github.com/Q-Future/Q-Instruct).

