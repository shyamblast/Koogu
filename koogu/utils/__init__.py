import os
import concurrent.futures
import queue
import logging
import platform
import tensorflow as tf


def instantiate_logging(log_file, log_level, cmdline_args, filemode='w'):

    os.makedirs(os.path.split(log_file)[0], exist_ok=True)

    logging.basicConfig(
        filename=log_file, level=log_level, filemode=filemode,
        format='%(asctime)s[%(levelname).1s] %(name)s: %(message)s',
        datefmt='%Y%m%dT%H%M%S')

    log_platform_info()
    logging.info('Command-line arguments: {}'.format(
        {k: v for k, v in vars(cmdline_args).items() if v is not None}))


def log_platform_info():

    logging.info(
        ('Platform: {:s} ({:s}, {:s} GPU). Python v{:s}. ' +
         'Tensorflow v{:s} ({:s} CUDA)').format(
            platform.node(),
            platform.platform(),
            'with' if tf.test.is_gpu_available(cuda_only=True) else 'no',
            platform.python_version(),
            tf.__version__,
            'with' if tf.test.is_built_with_cuda() else 'no'))


def processed_items_generator_mp(num_workers, processer_fn, raw_items,
                                 *processer_fn_args, **processer_fn_kwargs):
    """
    A generator for processing a sequence of inputs in a parallel fashion.
    :param processer_fn: is invoked on each item in :param raw_items:. This
    function 'yields' the results of one such invocation while other queued-up
    invocations run in the background.

    :param num_workers: The number of worker processes to spawn for parallel
        processing.
    :param processer_fn: A function handle.
    :param raw_items: An iterable (list, tuple, iterator) whose elements
        iteratively become input to processer_fn.
    :param processer_fn_args: Other non-keyword arguments passed as-is to
        processer_fn.
    :param processer_fn_kwargs: Other keyword arguments passed as-is to
        processer_fn.
    :return:
        Yields the tuple ('raw item', 'processing result') where 'raw item' is
        the element in :param raw_items: that was last processed and
        'processing result' is the result of applying :param processer_fn: to
        the element. Note that 'processing result' may be None (if
        :param processer_fn: does not return anything) or a tuple by itself (if
        :param processer_fn: returned a tuple).
    """

    assert num_workers > 0

    # Convert 'items' to an iterator if it isn't yet already
    if not (hasattr(raw_items, '__iter__') and hasattr(raw_items, '__next__')):
        if hasattr(raw_items, '__iter__'):
            raw_items = iter(raw_items)
        else:
            raw_items = iter([raw_items])

    logger = logging.getLogger(__name__)

    futures_queue = queue.Queue(num_workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) \
            as executor:

        # Add first num_workers items to the queue
        for idx, raw_item in enumerate(raw_items):
            futures_queue.put((raw_item,
                               executor.submit(processer_fn, raw_item,
                                               *processer_fn_args,
                                               **processer_fn_kwargs)))
            if idx == num_workers - 1:
                break

        while not futures_queue.empty():
            # Now wait for the result of the queue head
            head_raw_item, head_future = futures_queue.get()
            yield_curr_result = True
            try:
                head_results = head_future.result()
            except Exception as exc:
                # If item is list-like, use its first element as ID
                item_id = head_raw_item[0] if (
                        isinstance(head_raw_item, (list, tuple)) and
                        len(head_raw_item) > 0) else head_raw_item
                logger.error(f'Processing {item_id} generated an exception: ' +
                             repr(exc))
                yield_curr_result = False

            # Add another item to queue (if available).
            # Using a loop here just to avoid having to use a try-except block
            # around next(raw_items).
            for raw_item in raw_items:
                futures_queue.put((raw_item,
                                   executor.submit(processer_fn, raw_item,
                                                   *processer_fn_args,
                                                   **processer_fn_kwargs)))
                break

            if yield_curr_result:
                yield head_raw_item, head_results
