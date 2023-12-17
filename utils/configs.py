import datetime
import json
import logging
import os
import utils.constants as c
from pathlib import Path


def load_configs(path: str = None):
    root_dir =  os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
    standard_config_path = os.path.join(root_dir, "config.json")
    configs = None
    if path is None:
        path = standard_config_path

    try:
        with open(path, "r") as config_file_object:
            configs = json.load(config_file_object)

    except FileNotFoundError:
        logging.error("%s not found!", path)
        exit(1)

    return configs


def set_logging_parameters(
        opts: dict,
        release: bool
):
    if release:
        logging_level = logging.ERROR
    else:
        logging_level = logging.DEBUG

    log_file_dir, _ = get_storage_path(opts)
    log_file_name = log_file_dir / 'FSIM.log'

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if opts["general"]["write_logs"]:
        logging.basicConfig(
            filename=log_file_name,
            format='%(asctime)s : %(message)s',
            level=logging_level
        )
    else:
        logging.basicConfig(
            format='%(asctime)s : %(message)s',
            level=logging_level
        )


def get_storage_path(opts):
    base_path = opts["general"]["log_dir"] + '/logs'
    return get_unique_path(base_path, opts, mkdir=opts["general"]["write_logs"])


def get_unique_path(parent, opts, mkdir=True):
    """
    function to make a directory based on a unique path for logging info from time and configuration
    :param parent:
    :param opts:
    :param mkdir:
    :return:
    """
    if c.TIMESTAMP is None:
        parent = Path(parent)
        c.TIMESTAMP = datetime.datetime.now()
        current_dt = c.TIMESTAMP.strftime("%y%m%d%H%M%S")
        id_string = current_dt
        logger_path_candidate = parent / id_string
        folder_suffix_counter = 1
        if mkdir:
            while logger_path_candidate.exists():
                folder_suffix_counter += 1
                logger_path_candidate = parent / (id_string + str(folder_suffix_counter))
            logger_path_candidate.mkdir(parents=True, exist_ok=False)
            if folder_suffix_counter > 1:
                id_string = id_string + str(folder_suffix_counter)
        c.ID_STRING = id_string
        c.LOGGER_PATH_CANDIDATE = logger_path_candidate
    else:
        logger_path_candidate = c.LOGGER_PATH_CANDIDATE
        id_string = c.ID_STRING
    return logger_path_candidate, id_string
