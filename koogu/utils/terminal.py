import argparse


class ProgressBar:
    """
    Simple command-line ProgressBar.

    Example:
        >>> pbar = ProgressBar(300, prefix='Progress', length=60)
        >>> for idx in range(300):
        ...    sleep(.1)        # from time import sleep
        ...    pbar.increment()
        >>>
        >>> with ProgressBar(300, prefix='Progress', length=60, show_start=True) as pbar:
        ...     sleep(.8)
        ...     for idx in range(250):
        ...         sleep(.1)
        ...         pbar.increment()    # single increments
        ...
        ...     sleep(.5)
        ...     pbar.increment(50)    # bulk increments
    """

    def __init__(self, total_runs, prefix='', suffix='', decimals=1, length=80, fill='█', show_start=False):
        """

        :param total_runs: total iterations (integer)
        :param prefix: (optional) prefix string
        :param suffix: (optional) suffix string
        :param decimals: (optional) number of decimals in percent display (positive integer)
        :param length: (optional) character length of bar (integer)
        :param fill: (optional) bar fill character
        :param show_start: (optional) whether to display the starting (0.0%) state
        """

        self._total = total_runs
        self._prefix = prefix
        self._suffix = suffix
        self._length = length
        self._fill = fill

        self._curr_run = 0
        self._in_context = False
        self._decimal_part_fmt = "{0:" + str(4 + decimals) + "." + str(decimals) + "f}"

        if show_start:
            self._showbar()

    def __enter__(self):
        self._in_context = True
        return self

    def __exit__(self, *args):
        print()     # print new line when exiting context

    def _showbar(self):
        fraction = self._curr_run / float(self._total)
        filled_len = min(self._length, int(round(self._length * fraction)))

        print('\r{:s} |{:s}{:s}| {:s}% {:s}'.format(self._prefix,
                                                    self._fill * filled_len, '-' * (self._length - filled_len),
                                                    self._decimal_part_fmt.format(100. * fraction),
                                                    self._suffix), end='\r')

    def increment(self, incr=1):
        """Increment internal count and show the updated progress bar"""

        self._curr_run += incr

        self._showbar()

        if not self._in_context and self._curr_run == self._total:
            print()     # print new line at 100 %


class ArgparseConverters:
    """Abstract helper class providing functions for validation-cum-conversion of command-line parameters for argparse
    module."""

    @staticmethod
    def valid_percent(string):
        """A conversion-cum-validation helper to check that the given string value results in a percent value in the
         range [0 - 100]."""
        value = float(string)
        if value < 0. or value > 100.:
            raise argparse.ArgumentTypeError('%r is not a valid percent value' % string)
        return value

    @staticmethod
    def positive_integer(string):
        """A conversion-cum-validation helper to check that the given string value results in a positive integer."""
        value = int(string)
        if value <= 0:
            raise argparse.ArgumentTypeError('%r must be a positive integer value' % string)
        return value

    @staticmethod
    def positive_float(string):
        """A conversion-cum-validation helper to check that the given string value results in a positive float."""
        value = float(string)
        if value <= 0.:
            raise argparse.ArgumentTypeError('%r must be a positive quantity' % string)
        return value

    @staticmethod
    def float_0_to_1(string):
        """A conversion-cum-validation helper to check that the given string value results in range [0.0, 1.0]."""
        value = float(string)
        if value < 0.0 or value > 1.0:
            raise argparse.ArgumentTypeError('%r must be in the range [0.0, 1.0]' % string)
        return value

    @staticmethod
    def all_or_posint(string):
        """
        A conversion-cum-validation helper to check that the given string value is either a positive integer or 'all'.
        """
        if string.lower() == 'all':
            return 'all'

        return ArgparseConverters.positive_integer(string)
