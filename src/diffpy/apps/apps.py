import argparse

from diffpy.apps.app_runmacro import runmacro
from diffpy.apps.version import __version__  # noqa


class DiffpyHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Format subcommands without showing an extra placeholder entry."""

    def _format_action(self, action):
        if isinstance(action, argparse._SubParsersAction):
            return "".join(
                self._format_action(subaction)
                for subaction in self._iter_indented_subactions(action)
            )
        return super()._format_action(action)


def main():
    parser = argparse.ArgumentParser(
        prog="diffpy.apps",
        description=(
            "User applications to help with tasks using diffpy packages\n\n"
            "For more information, visit: "
            "https://github.com/diffpy/diffpy.apps/"
        ),
        formatter_class=DiffpyHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"diffpy.apps {__version__}",
    )
    apps_parsers = parser.add_subparsers(
        title="Available applications",
        dest="application",
    )
    runmacro_parser = apps_parsers.add_parser(
        "runmacro",
        help="Run a macro `<.dp-in>` file",
    )
    runmacro_parser.add_argument(
        "file",
        type=str,
        help="Path to the  `<.dp-in>` macro file to be run",
    )
    runmacro_parser.set_defaults(func=runmacro)
    args = parser.parse_args()
    if args.application is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
