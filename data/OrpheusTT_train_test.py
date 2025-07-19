import os
from huggingface_hub import login
from datasets import load_dataset, load_from_disk
from collections import Counter


# which language of the datasets do we need
name = "german"
# path to the repo do download the data from
path = "facebook/multilingual_librispeech"
# load_from_disk bool
load_disk_data = True

load_disk_data_path = "res/facebook_german/"
# maximal cores to use for preprocessing
num_cores = os.cpu_count()


if not load_disk_data:
    german_train = load_dataset(path = path, name = name, split = "train", streaming = False)
    german_test = load_dataset(path = path, name = name, split = "test", streaming = False)

    # declare the minimal count each id should at least have, since orpheusTTS depends on a larger size of avialabe data for a spefici speaker
    # for testing it is alright to use a smaller sample size, since both sets dont have any common speakers
    min_count_train = 10_000 
    min_count_test = 100

    client_counts_train = Counter(german_train["speaker_id"])   
    client_counts_test = Counter(german_test["speaker_id"])

    valid_clients_train = {speaker_id for speaker_id, count in client_counts_train.items() if count >= min_count_train}
    valid_clients_test = {speaker_id for speaker_id, count in client_counts_test.items() if count >= min_count_test}

    german_train_filter = german_train.filter(lambda example:example["speaker_id"] in valid_clients_train, num_proc = num_cores//7)
    german_test_filter = german_test.filter(lambda example:example["speaker_id"] in valid_clients_test, num_proc = num_cores//7)
else:
    german_train = load_from_disk(load_disk_data_path + "train/")
    german_test = load_from_disk(load_disk_data_path + "test/")
    
    final_dataset_train = german_train.map(lambda example: {"text" : example["transcript"],
                                                        "audio": example["audio"]["array"],
                                                        "sampling_rate": example["audio"]["sampling_rate"],
                                                        "audio_duration": example["audio_duration"],
                                                        "speaker_id": example["speaker_id"]}, num_proc = num_cores // 7,
                                                        remove_columns = german_train.column_names)
    
    final_dataset_test = german_test.map(lambda example:{"text" : example["transcript"],
                                                        "audio": example["audio"]["array"],
                                                        "sampling_rate": example["audio"]["sampling_rate"],
                                                        "audio_duration": example["audio_duration"],
                                                        "speaker_id": example["speaker_id"]}, num_proc = num_cores // 7,
                                                        remove_columns = german_test.column_names)
    
    login()
    
    final_dataset_train.push_to_hub("DenisDiCaprio/orpheutsTTS_facebook_german_train", private=True)
    final_dataset_test.push_to_hub("DenisDiCaprio/orpheutsTTS_facebook_german_test", private=True)