conda create -n dlo python=3.7.13
conda activate dlo
conda install cudatoolkit=11.3 -c pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install chardet typing_extensions==4.2.0 requests tqdm transformers==4.14.1
pip install pytorch_lightning==0.8.1 numpy==1.21.2 sentencepiece==0.1.96 scikit-learn send2trash
