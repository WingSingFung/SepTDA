import collections.abc
from pathlib import Path
from typing import Dict, Union

import numpy as np
from typeguard import check_argument_types
import json
import yaml


class NpyMSVSJsonWriter:
    """Writer class for a json file of numpy file.

    Examples:
        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...

        >>> writer = NpyMSVSJsonWriter('./data/', './data/feat.scp')
        >>> writer['aa'] = numpy_array
        >>> writer['bb'] = numpy_array

    """

    def __init__(self, outdir: Union[Path, str], scpfile: Union[Path, str]):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")

        self.data = {}

    def get_path(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        assert isinstance(value, np.ndarray), type(value)
        p = self.dir / f"{key}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), value)
        self.fscp.write(f"{key} {p}\n")

        # Store the file path
        self.data[key] = str(p)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()


def read_msvs_json_from_yaml(path: Union[Path, str], read_type: str = 'text') -> Dict[str, str]:
    # separate by #
    data_type, path = path.split('#', 1) if '#' in path else ("train", path)
    
    if data_type == "train":
        with open(path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        datasetsdirpath = Path(config_data["datasetsdirpath"])
        singing_datasets = config_data["singing_datasets"]
        return read_msvs_json_train(datasetsdirpath, singing_datasets, read_type)
    elif data_type == "valid":
        with open(path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        datasetsdirpath = Path(config_data["datasetsdirpath"])
        valid_datasets = config_data["valid_datasets"]
        return read_msvs_json_valid(datasetsdirpath, valid_datasets, read_type)
    elif data_type == "test":
        if '#' in path:
            test_data_path, yaml_path = path.split('#', 1)
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            test_data_path = config_data["test_datasets"]
        datasetsdirpath = Path(config_data["datasetsdirpath"])
        return read_msvs_json_test(datasetsdirpath, test_data_path, read_type)
    
def read_msvs_json_train(datasetsdirpath: Union[Path, str], singing_datasets: list, read_type: str = 'text') -> Dict[str, str]:
    if isinstance(datasetsdirpath, str):
        datasetsdirpath = Path(datasetsdirpath)
    data = {}
    for singing_dataset in singing_datasets:
        dataset_json_path = datasetsdirpath / singing_dataset / "train.json"
        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        for i, utt_data in enumerate(metadata):
            duration = float(utt_data["Duration"])
            if duration < 2.0:
                continue
            if read_type == "text":
                k, v = utt_data['Uid'], ""
            elif read_type == "npy":
                k, v = utt_data['Uid'], str(datasetsdirpath / singing_dataset / "audios" / f"{utt_data['Uid']}.npy")
            elif read_type == "wav":
                k, v = utt_data['Uid'], utt_data["Path"]

            if k in data:
                raise RuntimeError(f"{k} is duplicated ({dataset_json_path}:{utt_data['Uid']})")
            
            data[k] = v
    return data
    
def read_msvs_json_valid(datasetsdirpath: Union[Path, str], valid_datasets: list, read_type: str = 'text') -> Dict[str, str]:
    if isinstance(datasetsdirpath, str):
        datasetsdirpath = Path(datasetsdirpath)
        
    valid_datasets_json_paths = []
    for valid_dataset in valid_datasets:
        valid_datasets_json_paths.extend((datasetsdirpath / valid_dataset).glob("valid*.json"))
    
    data = {}
    for valid_datasets_json_path in valid_datasets_json_paths:
        with open(valid_datasets_json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        for i, utt_data in enumerate(metadata):
            data_type_uid = f"{utt_data['type']}_{utt_data['Uid']}"
            if read_type == "text":
                k, v = data_type_uid, ""
            elif read_type == "dict":
                k, v = data_type_uid, utt_data
            if data_type_uid in data:
                raise RuntimeError(f"{data_type_uid} is duplicated ({valid_datasets_json_path}:{utt_data['Uid']})")
            
            data[data_type_uid] = v
    return data

def read_msvs_json_test(datasetsdirpath: Union[Path, str], test_data: Union[list, Path, str], read_type: str = 'text') -> Dict[str, str]:
    if isinstance(datasetsdirpath, str):
        datasetsdirpath = Path(datasetsdirpath)
    
    test_data_jsons = []
    if isinstance(test_data, list):
        for i, test_dataset in enumerate(test_data):
            test_data_json = datasetsdirpath / test_dataset / "test.json"
            if test_data_json.exists():
                test_data_jsons.append(test_data_json)
    else:
        if isinstance(test_data, str):
            test_data = Path(test_data)
        test_data_jsons.append(test_data)
    
    data = {}
    for test_data_json in test_data_jsons:
        with open(test_data_json, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        for i, utt_data in enumerate(metadata):
            uid = utt_data['Uid']
            if read_type == "text":
                k, v = uid, ""
            elif read_type == "dict":
                k, v = uid, utt_data
            if uid in data:
                raise RuntimeError(f"{uid} is duplicated ({test_data_json}:{utt_data['Uid']})")

            data[uid] = v
    
    return data
    
class NpyMSVSJsonReader(collections.abc.Mapping):
    """Reader class for a scp file of numpy file.
    Examples:
        "\n\n"
        "   [    "
        "   {'Uid': 'utterance_id_A', ...},"
        "   {'Uid': 'utterance_id_B', ...},"
        "...]"
        >>> reader = NpyMSVSJsonReader('npy.json')
    """
    def __init__(self, fname: Union[Path, str]):
        assert check_argument_types()
        self.fname = Path(fname)
        self.data = read_msvs_json(fname, read_type='npy')
        
    def get_path(self, key):
        return self.data[key]

    def __getitem__(self, key) -> np.ndarray:
        p = self.data[key]
        return np.load(p)

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()