# Runs train.py with the parameters found during
# hyperparameter optimization

# Activate virtual environment
# cd ..
# source bin/activate 
# cd bsc-thesis-bdenboef

# Run the model

# Earthy sweep
# python3 train.py --train-batch-size 32 --eval-batch-size 8 --epochs 40 --weight-decay 0 --warmup-steps 15 --logging-steps 10 --learning-rate "3e-4"

# Swept sweep
python3 train.py --train-batch-size 4 --eval-batch-size 32 --epochs 50 --weight-decay 0 --warmup-steps 5 --logging-steps 10 --learning-rate "8e-5"

# Ruby sweep
# python3 train.py --train-batch-size 4 --eval-batch-size 16 --epochs 50 --weight-decay 0 --warmup-steps 10 --logging-steps 10 --learning-rate "6e-5"


