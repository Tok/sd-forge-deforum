# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import random

def next_seed(args, root):
    if args.seed_behavior == 'iter':
        args.seed += 1 if root.seed_internal % args.seed_iter_N == 0 else 0
        root.seed_internal += 1
    elif args.seed_behavior == 'ladder':
        args.seed += 2 if root.seed_internal == 0 else -1
        root.seed_internal = 1 if root.seed_internal == 0 else 0
    elif args.seed_behavior == 'alternate':
        args.seed += 1 if root.seed_internal == 0 else -1
        root.seed_internal = 1 if root.seed_internal == 0 else 0
    elif args.seed_behavior == 'fixed':
        pass  # always keep seed the same
    else:
        args.seed = random.randint(0, 2**32 - 1)
    return args.seed


def generate_next_seed(args, seed, seed_control: int = 0):
    # Same as 'next_seed' but adjusted for asynchronous usage from experimental render core.
    # Side effects not required (except for the call to randint), because all seeds are generated before the generation.
    def _increment_after_n():
        return 1 if seed_control % args.seed_iter_N == 0 else 0

    def _two_up_or_one_down():
        return 2 if seed_control == 0 else -1

    def _one_up_or_down():
        return 1 if seed_control == 0 else -1

    def _alternate_control():
        return 1 if seed_control == 0 else 0

    if args.seed_behavior == 'iter':
        return seed + _increment_after_n(), seed_control + 1
    elif args.seed_behavior == 'ladder':
        return seed + _two_up_or_one_down(), _alternate_control()
    elif args.seed_behavior == 'alternate':
        return seed + _one_up_or_down(), _alternate_control()
    elif args.seed_behavior == 'fixed':
        return seed, seed_control  # always keep seed the same
    else:
        return random.randint(0, 2**32 - 1), seed_control
