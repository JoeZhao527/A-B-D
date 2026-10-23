"""All parameters of one factor-analysis run, with defaults, as one JSON-able object."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields


@dataclass
class ParcelConfig:
    enabled: bool = False
    r_threshold: float = 0.8        # every pair inside a parcel must correlate >= this (complete linkage)
    linkage: str = "complete"       # complete | average
    scope: str = "all"              # all | within_family   (family = prefix before ':' in the trait name)
    align_sign: bool = True         # flip members whose first-PC loading is negative before averaging
    min_pc1: float = 0.70           # unidimensionality: first-PC share of a parcel must reach this
    min_alpha: float = 0.80         # ... and Cronbach's alpha this; otherwise the cluster is not merged


@dataclass
class RetentionConfig:
    rule: str = "parallel"          # parallel | kaiser | map | fixed
    pa_percentile: float = 95.0     # Glorfeld's conservative variant; 50 = Horn's mean
    pa_n_random: int = 500
    pa_basis: str = "pca"           # pca (eigenvalues of R) | smc (reduced R with SMC diagonal)
    pa_data: str = "normal"         # normal | permute (column-wise permutation of the observed data)
    map_power: int = 2              # Velicer 1976 (2) or the 2000 revision (4)
    k_fixed: int | None = None      # used when rule == fixed
    k_min: int = 1
    k_max: int | None = None        # default: p // 2


@dataclass
class ExtractConfig:
    method: str = "paf"             # principal-axis factoring with iterated communalities
    max_iter: int = 100
    tol: float = 1e-6


@dataclass
class RotateConfig:
    method: str = "varimax"         # varimax | promax | none
    promax_power: int = 4
    kaiser_normalize: bool = True


@dataclass
class ScoresConfig:
    method: str = "regression"      # Thurstone regression scores
    standardize: bool = True
    ridge: float = 0.1              # W = (R + ridge I)^-1 S; 0 = plain inverse


@dataclass
class FAConfig:
    parcel: ParcelConfig = field(default_factory=ParcelConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    rotate: RotateConfig = field(default_factory=RotateConfig)
    scores: ScoresConfig = field(default_factory=ScoresConfig)
    seed: int = 0
    loading_cutoff: float = 0.40    # reporting only: which loadings the summary lists
    drop_models_with_nan: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), indent=2, **kw)

    @classmethod
    def from_dict(cls, d: dict) -> "FAConfig":
        sub = {"parcel": ParcelConfig, "retention": RetentionConfig, "extract": ExtractConfig,
               "rotate": RotateConfig, "scores": ScoresConfig}
        kw = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            kw[f.name] = sub[f.name](**d[f.name]) if f.name in sub else d[f.name]
        return cls(**kw)

    @classmethod
    def from_json(cls, s: str) -> "FAConfig":
        return cls.from_dict(json.loads(s))
