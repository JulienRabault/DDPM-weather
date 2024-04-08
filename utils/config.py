import json
import logging
import os
from pathlib import Path
import jsonschema as jsonschema
import yaml

from utils.distributed import is_main_gpu, get_rank_num, synchronize
import datetime

CONFIG_SCHEMA_PATH = "utils/config_schema.json"
DATASET_CONFIG_SCHEMA_PATH = "utils/dataset_config_schema.json"


def load_yaml(yaml_path):
    with open(yaml_path, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data


TYPE_MAPPER = {
    "string": str,
    "integer": int,
    "int": int,
    "str": str,
    "boolean": bool,
    "array": list,
    "list": list,
    "float": float,
    "dict": dict,
    "number": float,
}


class Config:
    def __init__(self, args, schema_path, modified_args):
        # Load YAML configuration file and initialize logger
        self.schema_path = schema_path
        # default values of the config
        self._update_from_args(args)
        self._update_from_config(args.yaml_path, overload=modified_args[::2])

        self.logger = logging.getLogger(f"logddp_{get_rank_num()}")

        self._validate_config()
        self.basename = self.run_name

    def __str__(self):
        # Return a string representation of the configuration
        config_string = "Configuration:"
        for attr, value in self.to_dict().items():
            config_string += f"\n\t{attr}: {value}"
        config_string += (
            f"\n\tWANDB_MODE: {os.environ.get('WANDB_MODE', 'Not set')}\n"
        )
        return config_string

    def _update_from_args(self, args):
        # Update configuration attributes from command line arguments
        logging.debug(
            f"Updating configuration from command line arguments: {args}"
        )
        logging.debug(f"Schema path: {self.schema_path}")
        for prop, value in args.__dict__.items():
            setattr(self, prop, value)

    def _update_from_dict(self, dict):
        # Update configuration attributes from dict
        for prop, value in dict.items():
            setattr(self, prop, value)

    def _validate_config(self):
        # Validate the configuration against a JSON schema
        with open(self.schema_path, "r") as schema_file:
            schema = json.load(schema_file)
        jsonschema.validate(self.__dict__, schema)
        # Check specific conditions for certain configuration values
        self.output_dir = Path(self.output_dir)
        if (
            self.sampling_mode == "guided"
            or self.sampling_mode == "simple_guided"
        ):
            assert (
                self.guidance_loss_scale >= 0
                and self.guidance_loss_scale <= 100
            ), "Guidance loss scale must be between 0 and 100."
            if self.data_dir is None:
                raise ValueError(
                    "data_dir must be specified when using guided sampling mode."
                )
            if (
                self.mode != "Sample"
                and self.sampling_mode == "guided"
                and self.guiding_col is None
            ):
                raise ValueError(
                    "guiding_col must be specified when using guided sampling mode."
                )
        if self.resume:
            if self.model_path is None or not os.path.isfile(self.model_path):
                raise FileNotFoundError(
                    f"self.resume={self.resume} but snapshot_path={self.model_path} is None or doesn't exist"
                )
            if self.mode != "Train":
                raise ValueError("--r flag can only be used in Train mode.")
        if self.any_time > self.epochs:
            if is_main_gpu():
                self.logger.warning(
                    f"any_time={self.any_time} is greater than epochs={self.epochs}. "
                )
        if "rr" in self.var_indexes:
            if self.dataset_config_file is None:
                raise ValueError(
                    "field dataset_config_file should not be None / should be spec'd if rr is among the "
                    "variables"
                )
        if self.dataset_config_file is not None and (
            self.mean_file is not None or self.max_file is not None
        ):
            raise ValueError(
                "mean_file and max_file should not be specified if dataset_config_file is specified, "
                "and vice versa"
            )
        cond_n_sample = (
            self.batch_size
            if isinstance(self.batch_size, int)
            else min(self.batch_size)
        )

        if self.n_sample > cond_n_sample and self.guiding_col is not None:
            self.n_sample = cond_n_sample
            if is_main_gpu():
                self.logger.warning(
                    f"n_sample={self.n_sample} is greater than batch_size={self.batch_size}. "
                    f"Setting n_sample={self.n_sample} to batch_size={self.batch_size}."
                )
        # Check and create directories based on the configuration
        paths = [
            self.output_dir / self.run_name,
            self.output_dir / self.run_name / "samples",
        ]
        if self.mode == "Train":
            paths.append(self.output_dir / self.run_name / "WANDB")
            paths.append(self.output_dir / self.run_name / "WANDB/cache")
        self._next_run_dir(paths)

        return

    def to_dict(self):
        # Convert configuration to a dictionary
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }

    def to_json(self):
        # Convert configuration to a JSON string
        return json.dumps(self.to_dict())

    def to_yaml(self):
        # Convert configuration to a YAML string
        return yaml.dump(self.to_dict())

    def save(self, path):
        # Save configuration to a YAML file
        with open(path, "w+") as f:
            yaml.dump(self.to_dict(), f)

    @classmethod
    def from_args_and_yaml(cls, args, modified_args):
        # Create a Config object from command line arguments and a YAML file
        config = cls(args, CONFIG_SCHEMA_PATH, modified_args)
        return config

    @classmethod
    def create_arguments(cls, parser):
        # Dynamically create argparse arguments based on the schema
        with open(CONFIG_SCHEMA_PATH, "r") as schema_file:
            schema = json.load(schema_file)

        for prop, prop_schema in schema["properties"].items():
            try:
                arg_type = TYPE_MAPPER.get(prop_schema.get("type", "str"), str)
            except:
                arg_type = TYPE_MAPPER.get(
                    prop_schema.get("type", "str")[0], str
                )
            arg_default = prop_schema.get("default", None)
            arg_help = prop_schema.get("description", None)
            if arg_type == list:
                parser.add_argument(
                    f"--{prop}",
                    nargs="+",
                    type=arg_type,
                    default=arg_default,
                    help=arg_help,
                )
            elif arg_type == bool:
                parser.add_argument(
                    f"--{prop}",
                    default=arg_default,
                    help=arg_help,
                    action="store_true",
                )
            else:
                parser.add_argument(
                    f"--{prop}",
                    type=arg_type,
                    default=arg_default,
                    help=arg_help,
                )

    def _next_run_dir(self, paths, suffix=None):
        # Create directories for the next run
        if self.resume:
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"The following directories do not exist: {path}"
                    )

        else:
            current_datetime = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S%f"
            )[:-3]
            train_num = 1
            train_name = self.run_name
            while os.path.exists(train_name):
                if suffix is not None:
                    train_name = (
                        self.basename + "__" + suffix + "_" + current_datetime
                    )
                else:
                    while os.path.exists(train_name):
                        if f"_{train_num}" in train_name:
                            train_name = (
                                "_".join(train_name.split("_")[:-1])
                                + f"_{train_num + 1}"
                            )
                            train_num += 1
                        else:
                            train_name = f"{train_name}_{train_num}"

            self.run_name = train_name

            paths = [
                self.output_dir / self.run_name,
                self.output_dir / self.run_name / "samples",
            ]
            if self.mode == "Train":
                paths.append(self.output_dir / self.run_name / "WANDB/")
                paths.append(self.output_dir / self.run_name / "WANDB/cache")
            synchronize()
            for path in paths:
                os.makedirs(path, exist_ok=True)

    def _update_from_config(self, yaml_path, overload: list):
        yaml_config = load_yaml(yaml_path)
        overload = [s.lstrip("-") for s in overload]
        for key, value in yaml_config.items():
            if key not in overload:
                setattr(self, key, value)
            else:
                logging.warning(f"Overloading {key} to {getattr(self, key)}")


class DataSetConfig(Config):
    def __init__(self, yaml_path):
        # Load YAML configuration file and initialize logger
        yaml_config = load_yaml(yaml_path)
        for prop, value in yaml_config.items():
            setattr(self, prop, value)

        self._validate_config()

    def _validate_config(self):
        # Validate the configuration against a JSON schema
        with open(DATASET_CONFIG_SCHEMA_PATH, "r") as schema_file:
            schema = json.load(schema_file)
        jsonschema.validate(self.__dict__, schema)
