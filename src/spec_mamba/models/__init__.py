from typing import TypeAlias, TypeVar

from spec_mamba.models.audio_mamba import *
from spec_mamba.models.ssast import *

ModelType: TypeAlias = AudioMamba | AudioMambaCLF | SSAST | SSASTCLF
ModelTypeVar = TypeVar("ModelTypeVar", bound=ModelType)
