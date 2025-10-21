# quick start
```bash
# install quant package
pip uninstall datasets modelscope -y
pip install addict 
pip install zstandard
pip install "modelscope[dataset]" -upgrade
pip install -e .
# run awq example, prepare model in advance
cd quant/examples
python awq_quantize.py
```
