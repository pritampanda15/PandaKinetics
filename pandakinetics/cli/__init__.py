"""PandaKinetics CLI Module - Robust Import Handling"""

import sys
import logging
logger = logging.getLogger(__name__)

# Try multiple import strategies
main = None
cli = None

# Strategy 1: Import from cli.py
try:
    from pandakinetics.cli.cli import main
    try:
        from pandakinetics.cli.cli import cli
    except ImportError:
        pass
    logger.debug("Imported from cli.py")
except ImportError:
    # Strategy 2: Import from main.py
    try:
        from pandakinetics.cli.main import main
        try:
            from pandakinetics.cli.main import cli
        except ImportError:
            pass
        logger.debug("Imported from main.py")
    except ImportError:
        # Strategy 3: Create minimal working CLI
        import click
        
        @click.group()
        @click.option('-v', '--verbose', is_flag=True, help='Enable verbose logging')
        @click.option('--gpu', help='Specify GPU device')
        @click.pass_context
        def cli(ctx, verbose, gpu):
            """PandaKinetics: Multi-Scale Structure-Kinetics Simulator"""
            ctx.ensure_object(dict)
            ctx.obj['verbose'] = verbose
            if verbose:
                logging.basicConfig(level=logging.DEBUG)
        
        # Add available commands
        try:
            from pandakinetics.cli.commands.predict import predict
            cli.add_command(predict)
        except ImportError:
            pass
        try:
            from pandakinetics.cli.commands.benchmark import benchmark
            cli.add_command(benchmark)
        except ImportError:
            pass
        try:
            from pandakinetics.cli.commands.visualize import visualize
            cli.add_command(visualize)
        except ImportError:
            pass
        try:
            from pandakinetics.cli.commands.validate import validate_config
            cli.add_command(validate_config)
        except ImportError:
            pass
        
        def main():
            try:
                cli()
            except Exception as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)

__all__ = ['main', 'cli']
