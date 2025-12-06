"""
Configuration loading and validation for OFI market making strategies.

Loads YAML config files and validates parameters before backtest execution.
All parameters are pre-defined (no optimization allowed).
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class StrategyConfig:
    """Complete strategy configuration loaded from YAML."""
    
    # Strategy metadata
    name: str
    description: str
    version: str
    
    # Quoting parameters
    risk_aversion: float
    terminal_time: float
    order_arrival_rate: float
    max_inventory: int
    min_inventory: int
    tick_size: float
    min_spread_bps: float
    signal_adjustment_factor: float
    inventory_urgency_factor: float
    
    # Feature parameters
    ofi_beta: float
    ofi_horizon_seconds: int
    volatility_halflife_seconds: float
    volatility_min_periods: int
    alpha_ofi: float
    alpha_imbalance: float
    
    # Fill model parameters
    base_intensity: float
    decay_rate: float
    min_probability: float
    max_probability: float
    
    # Backtest parameters
    initial_cash: float
    initial_inventory: int
    commission_per_share: float
    random_seed: int
    
    # Data parameters
    start_time: str
    end_time: str
    resample_frequency: str
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_quoting_params()
        self._validate_feature_params()
        self._validate_fill_params()
        self._validate_backtest_params()
    
    def _validate_quoting_params(self):
        """Validate quoting parameters."""
        if self.risk_aversion <= 0:
            raise ValueError(f"risk_aversion must be > 0, got {self.risk_aversion}")
        if self.terminal_time <= 0:
            raise ValueError(f"terminal_time must be > 0, got {self.terminal_time}")
        if self.order_arrival_rate <= 0:
            raise ValueError(f"order_arrival_rate must be > 0, got {self.order_arrival_rate}")
        if self.max_inventory <= 0:
            raise ValueError(f"max_inventory must be > 0, got {self.max_inventory}")
        if self.min_inventory >= 0:
            raise ValueError(f"min_inventory must be < 0, got {self.min_inventory}")
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be > 0, got {self.tick_size}")
        if self.min_spread_bps < 0:
            raise ValueError(f"min_spread_bps must be >= 0, got {self.min_spread_bps}")
        if self.signal_adjustment_factor < 0:
            raise ValueError(f"signal_adjustment_factor must be >= 0, got {self.signal_adjustment_factor}")
        if self.inventory_urgency_factor < 1:
            raise ValueError(f"inventory_urgency_factor must be >= 1, got {self.inventory_urgency_factor}")
    
    def _validate_feature_params(self):
        """Validate feature engineering parameters."""
        # OFI beta should be positive (from replication study)
        if self.ofi_beta <= 0:
            raise ValueError(f"ofi_beta must be > 0, got {self.ofi_beta}")
        if self.ofi_horizon_seconds <= 0:
            raise ValueError(f"ofi_horizon_seconds must be > 0, got {self.ofi_horizon_seconds}")
        
        # Volatility parameters
        if self.volatility_halflife_seconds <= 0:
            raise ValueError(f"volatility_halflife_seconds must be > 0, got {self.volatility_halflife_seconds}")
        if self.volatility_min_periods <= 0:
            raise ValueError(f"volatility_min_periods must be > 0, got {self.volatility_min_periods}")
        
        # Signal blending weights (can be 0 for baseline strategies)
        if self.alpha_ofi < 0 or self.alpha_ofi > 1:
            raise ValueError(f"alpha_ofi must be in [0, 1], got {self.alpha_ofi}")
        if self.alpha_imbalance < 0 or self.alpha_imbalance > 1:
            raise ValueError(f"alpha_imbalance must be in [0, 1], got {self.alpha_imbalance}")
    
    def _validate_fill_params(self):
        """Validate fill model parameters."""
        if self.base_intensity <= 0:
            raise ValueError(f"base_intensity must be > 0, got {self.base_intensity}")
        if self.decay_rate <= 0:
            raise ValueError(f"decay_rate must be > 0, got {self.decay_rate}")
        if self.min_probability <= 0 or self.min_probability >= 1:
            raise ValueError(f"min_probability must be in (0, 1), got {self.min_probability}")
        if self.max_probability <= 0 or self.max_probability >= 1:
            raise ValueError(f"max_probability must be in (0, 1), got {self.max_probability}")
        if self.min_probability >= self.max_probability:
            raise ValueError(f"min_probability must be < max_probability")
    
    def _validate_backtest_params(self):
        """Validate backtest parameters."""
        if self.commission_per_share < 0:
            raise ValueError(f"commission_per_share must be >= 0, got {self.commission_per_share}")
        if self.random_seed < 0:
            raise ValueError(f"random_seed must be >= 0, got {self.random_seed}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'quoting': {
                'risk_aversion': self.risk_aversion,
                'terminal_time': self.terminal_time,
                'order_arrival_rate': self.order_arrival_rate,
                'max_inventory': self.max_inventory,
                'min_inventory': self.min_inventory,
                'tick_size': self.tick_size,
                'min_spread_bps': self.min_spread_bps,
                'signal_adjustment_factor': self.signal_adjustment_factor,
                'inventory_urgency_factor': self.inventory_urgency_factor,
            },
            'features': {
                'ofi_beta': self.ofi_beta,
                'ofi_horizon_seconds': self.ofi_horizon_seconds,
                'volatility_halflife_seconds': self.volatility_halflife_seconds,
                'volatility_min_periods': self.volatility_min_periods,
                'alpha_ofi': self.alpha_ofi,
                'alpha_imbalance': self.alpha_imbalance,
            },
            'fill_model': {
                'base_intensity': self.base_intensity,
                'decay_rate': self.decay_rate,
                'min_probability': self.min_probability,
                'max_probability': self.max_probability,
            },
            'backtest': {
                'initial_cash': self.initial_cash,
                'initial_inventory': self.initial_inventory,
                'commission_per_share': self.commission_per_share,
                'random_seed': self.random_seed,
            },
            'data': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'resample_frequency': self.resample_frequency,
            },
            'metadata': self.metadata,
        }


def load_config(config_path: str) -> StrategyConfig:
    """
    Load strategy configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML config file
    
    Returns
    -------
    StrategyConfig
        Validated configuration object
    
    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    ValueError
        If config is invalid or missing required fields
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Extract nested parameters
    strategy = raw_config.get('strategy', {})
    quoting = raw_config.get('quoting', {})
    features = raw_config.get('features', {})
    fill_model = raw_config.get('fill_model', {})
    backtest = raw_config.get('backtest', {})
    data = raw_config.get('data', {})
    metadata = raw_config.get('metadata', {})
    
    # Build config object
    try:
        config = StrategyConfig(
            # Strategy metadata
            name=strategy['name'],
            description=strategy['description'],
            version=strategy['version'],
            
            # Quoting parameters
            risk_aversion=float(quoting['risk_aversion']),
            terminal_time=float(quoting['terminal_time']),
            order_arrival_rate=float(quoting['order_arrival_rate']),
            max_inventory=int(quoting['max_inventory']),
            min_inventory=int(quoting['min_inventory']),
            tick_size=float(quoting['tick_size']),
            min_spread_bps=float(quoting['min_spread_bps']),
            signal_adjustment_factor=float(quoting['signal_adjustment_factor']),
            inventory_urgency_factor=float(quoting['inventory_urgency_factor']),
            
            # Feature parameters
            ofi_beta=float(features['ofi_beta']),
            ofi_horizon_seconds=int(features['ofi_horizon_seconds']),
            volatility_halflife_seconds=float(features['volatility_halflife_seconds']),
            volatility_min_periods=int(features['volatility_min_periods']),
            alpha_ofi=float(features['alpha_ofi']),
            alpha_imbalance=float(features['alpha_imbalance']),
            
            # Fill model parameters
            base_intensity=float(fill_model['base_intensity']),
            decay_rate=float(fill_model['decay_rate']),
            min_probability=float(fill_model['min_probability']),
            max_probability=float(fill_model['max_probability']),
            
            # Backtest parameters
            initial_cash=float(backtest['initial_cash']),
            initial_inventory=int(backtest['initial_inventory']),
            commission_per_share=float(backtest['commission_per_share']),
            random_seed=int(backtest['random_seed']),
            
            # Data parameters
            start_time=str(data['start_time']),
            end_time=str(data['end_time']),
            resample_frequency=str(data['resample_frequency']),
            
            # Metadata
            metadata=metadata,
        )
    except KeyError as e:
        raise ValueError(f"Missing required config field: {e}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid config value: {e}")
    
    return config


def list_available_configs(config_dir: str = "configs") -> list[str]:
    """
    List all available strategy configurations.
    
    Parameters
    ----------
    config_dir : str
        Directory containing config files
    
    Returns
    -------
    list[str]
        List of config file names (without .yaml extension)
    """
    config_path = Path(config_dir)
    if not config_path.exists():
        return []
    
    configs = []
    for file in config_path.glob("*.yaml"):
        configs.append(file.stem)
    
    return sorted(configs)


def validate_all_configs(config_dir: str = "configs") -> Dict[str, bool]:
    """
    Validate all configuration files in directory.
    
    Parameters
    ----------
    config_dir : str
        Directory containing config files
    
    Returns
    -------
    Dict[str, bool]
        Dictionary mapping config name to validation status
    """
    results = {}
    for config_name in list_available_configs(config_dir):
        config_path = Path(config_dir) / f"{config_name}.yaml"
        try:
            load_config(str(config_path))
            results[config_name] = True
        except Exception as e:
            print(f"❌ {config_name}: {e}")
            results[config_name] = False
    
    return results


if __name__ == "__main__":
    """Test config loading."""
    import sys
    
    # Validate all configs
    print("Validating all strategy configurations...")
    results = validate_all_configs()
    
    print(f"\n{'='*60}")
    print(f"Configuration Validation Results")
    print(f"{'='*60}")
    
    for config_name, valid in results.items():
        status = "✅ VALID" if valid else "❌ INVALID"
        print(f"{status:12s} {config_name}")
    
    # Load and display each config
    for config_name in list_available_configs():
        if results[config_name]:
            print(f"\n{'-'*60}")
            print(f"Configuration: {config_name}")
            print(f"{'-'*60}")
            
            config = load_config(f"configs/{config_name}.yaml")
            print(f"Name: {config.name}")
            print(f"Description: {config.description}")
            print(f"OFI Signal Weight (α_ofi): {config.alpha_ofi}")
            print(f"Signal Adjustment Factor: {config.signal_adjustment_factor}")
            print(f"Risk Aversion (γ): {config.risk_aversion}")
            print(f"Anti-Overfitting: {config.metadata.get('anti_overfitting', 'N/A')}")
    
    print(f"\n{'='*60}")
    print(f"✅ All {len([v for v in results.values() if v])}/{len(results)} configs valid")
    print(f"{'='*60}")
