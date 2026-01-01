# Fix the code to latest torch
sed -i \
  's/super(WarmupLambdaLR, self).__init__(optimizer, lr_lambda, last_epoch, verbose)/super(WarmupLambdaLR, self).__init__(optimizer, lr_lambda, last_epoch)/' \
  cosmos_predict1/utils/scheduler.py

# Changes to the DataLoader:
# - Enable shuffling of data, which could be beneficial if the files are ordered
# - Set num_workers to 1 for much lower memory usage compared to 8, and no memory explosion
sed -i \
  -e 's/shuffle=None/shuffle=True/' \
  -e 's/num_workers=num_workers/num_workers=1/' \
  cosmos_predict1/tokenizer/training/configs/base/data.py

# change supported resolutions to lower values
sed -i \
  -e 's/\["1080", "720", "480", "360", "256"\]/["256", "128", "64"]/' \
  cosmos_predict1/tokenizer/training/configs/registry.py \

