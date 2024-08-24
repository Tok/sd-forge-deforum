import random

from PIL import Image


def put_all(dictionaries, key, value):
    return list(map(lambda d: {**d, key: value}, dictionaries))


def put_if_present(dictionary, key, value):
    if value is not None:
        dictionary[key] = value


def _call_or_use(callable_or_value):
    return callable_or_value() if callable(callable_or_value) else callable_or_value


def call_or_use_on_cond(condition, callable_or_value):
    return _call_or_use(callable_or_value) if condition else None


def create_img(dimensions):
    return Image.new('1', dimensions, 1)


def generate_random_seed():
    return random.randint(0, 2 ** 32 - 1)
