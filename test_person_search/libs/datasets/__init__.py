import threading
import sys
if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue

import torch

from .cuhk_sysu import CUHK_SYSU
from .prw import PRW

from ..utils.transforms import get_transform
from ..utils.group_by_aspect_ratio import create_aspect_ratio_groups,\
    GroupedBatchSampler


class PrefetchGenerator(threading.Thread):

    def __init__(self, generator, max_prefetch=1):
        super(PrefetchGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class PrefetchDataLoader(torch.utils.data.DataLoader):

    def __iter__(self):
        return PrefetchGenerator(
            super(PrefetchDataLoader, self).__iter__()
        )


def collate_fn(x):
    return x


def get_dataset(dataset_target, train=True):
    paths = {
        'cuhk_det': ('/root/workplace/dataset/CUHK-SYSU/', CUHK_SYSU),
        'prw_det': ('/root/workplace/person_search/NAE4PS/data/PRW', PRW)
    }

    if dataset_target == 'prw':
        dataset_target = 'prw_det'

    if dataset_target == 'cuhk':
        dataset_target = 'cuhk_det'
    
    if dataset_target == 'cuhk2prw':
        dataset_target = 'prw_det'

    if dataset_target == 'prw2cuhk':
        dataset_target = 'cuhk_det'

    p, ds_cls = paths[dataset_target]
 
    test_set = ds_cls(p, get_transform(False),
                        mode='test')
    probe_set = ds_cls(p, get_transform(False),
                        mode='probe')
    return test_set, probe_set


def get_data_loader(dataset_target, train=True):
    dataset = get_dataset(dataset_target, train)

    test_sampler = torch.utils.data.SequentialSampler(dataset[0])
    probe_sampler = torch.utils.data.SequentialSampler(dataset[1])

    data_loader_test = PrefetchDataLoader(
        dataset[0], batch_size=1,
        sampler=test_sampler, num_workers=1,
        collate_fn=collate_fn)
    data_loader_probe = PrefetchDataLoader(
        dataset[1], batch_size=1,
        sampler=probe_sampler, num_workers=1,
        collate_fn=collate_fn)

    return data_loader_test, data_loader_probe
