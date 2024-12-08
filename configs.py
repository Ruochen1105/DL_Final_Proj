
from dataclasses import dataclass
from omegaconf import OmegaConf

@dataclass
class ConfigBase:
    @classmethod
    def parse_from_command_line(cls):
        return OmegaConf.structured(cls)

    @classmethod
    def parse_from_file(cls, path: str):
        oc = OmegaConf.load(path)
        return cls.parse_from_dict(OmegaConf.to_container(oc))

    @classmethod
    def parse_from_dict(cls, inputs):
        return OmegaConf.structured(cls(**inputs))

    def save(self, path: str):
        with open(path, "w") as f:
            OmegaConf.save(config=self, f=f)
