#!/usr/bin/env python3
import click
import logging
import sys

@click.group()
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose logging')
@click.option('--gpu', help='Specify GPU device (e.g., cuda:0)')
@click.pass_context
def cli(ctx, verbose, gpu):
    """PandaKinetics: Multi-Scale Structure-Kinetics Simulator for Drug Design"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['gpu'] = gpu
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

# Import and add commands
try:
    from pandakinetics.cli.commands.predict import predict
    cli.add_command(predict)
except ImportError as e:
    click.echo(f"Warning: Could not import predict command: {e}")

try:
    from pandakinetics.cli.commands.benchmark import benchmark
    cli.add_command(benchmark)
except ImportError as e:
    click.echo(f"Warning: Could not import benchmark command: {e}")

try:
    from pandakinetics.cli.commands.visualize import visualize
    cli.add_command(visualize)
except ImportError as e:
    click.echo(f"Warning: Could not import visualize command: {e}")

try:
    from pandakinetics.cli.commands.validate import validate_config
    cli.add_command(validate_config)
except ImportError as e:
    click.echo(f"Warning: Could not import validate-config command: {e}")

def main():
    """Main entry point for the CLI"""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n⚠️  Interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
