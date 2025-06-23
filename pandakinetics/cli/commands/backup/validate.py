#!/usr/bin/env python3
import click
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@click.command('validate-config')
@click.argument('config_file', type=click.Path(exists=True))
def validate_config(config_file):
    """Validate configuration file"""
    
    config_path = Path(config_file)
    
    try:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            click.echo("❌ Unsupported format. Use .yaml or .json")
            return 1
        
        click.echo("✅ Configuration file is valid!")
        return 0
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}")
        return 1
