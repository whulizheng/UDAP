# Official PyTorch implementation of "Universal Adversarial Purification with DDIM Metric Loss for Stable Diﬀusion" (AAAI'26)

## Todo

- [x] Release core code of UDAP
- [x] Release a simple demo with strongly adversarial images by PID (https://github.com/PKU-ML/Diffusion-PID-Protection/tree/main) and Anti-DB (https://github.com/VinAIResearch/Anti-DreamBooth)
- [ ] Please contact me or post issues if you have any problems

## Environment setup

Install dependencies:
```shell
cd UDAP
conda create -n UDAP python=3.9
conda activate UDAP
pip install -r requirements.txt
```

## How to run

To purify adversarial images from demo, you can run
```python
python Main.py
```


## Contacts
If you have any problems, please open an issue in this repository or send an email to [umlizheng@gmail.com](mailto:umlizheng@gmail.com).
