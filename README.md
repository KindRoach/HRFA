## Note

A large amount of disk space is required for storing processed data.

## Setup Environment

See package in ```requirements.txt```

## Prepare data

Download ```.json``` files from https://nijianmo.github.io/amazon/index.html and put them under path ```data```.

```
├───data
│       AmazonFashion.json
│       Appliances.json
│       DigitalMusic.json
│       ...
```

## Run Code

```
# Pre-process
python -m data_process.data_reader.process_raw_data
python -m data_process.data_reader.generate_train_data

# Train & Test
python train.py
python test.py
```
