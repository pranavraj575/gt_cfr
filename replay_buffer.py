"""
from werner-duvaud/muzero-general

config:
tensor_tuple -> whether all data will be a tuple of tensors. If true, will sample and batch automatically
capacity -> max capacity of buffer

"""
import copy
import time

import numpy as np
import torch
from collections import deque

import torch, os, shutil, pickle


class AbsReplayBuffer:
    def __init__(self, config, initial_buffer):
        self.config = config
        if initial_buffer is not None:
            self.extend(initial_buffer)

    def reset_storage(self):
        """
        resets internal buffer
        """
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.push(item)

    def push(self, item):
        """
        pushes an item into replay buffer
        Args:
            item: item
        Returns: item that is displaced, or None if no such item
        """
        raise NotImplementedError

    def sample_one(self):
        raise NotImplementedError

    def clear(self):
        pass

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass

    def get_all(self):
        raise NotImplementedError

    def sample(self, batch=None, **kwargs):
        if batch is None:
            batch = self.config.get('batch', 128)
        if self.config.get('tensor_tuple', False):
            stuff = [self.sample_one() for _ in range(batch)]
            # stuff=[(t11,t12,...,t1n),(t21,t22,...,t2n),...]
            # convert this to stack((t11,t21,t31,...)), stack((t12,t22,t32,...)), ...
            return tuple(torch.stack([s[i] for s in stuff]) for i in range(len(stuff[0])))
        else:
            return [self.sample_one() for _ in range(batch)]

    def __len__(self):
        raise NotImplementedError


class ReplayBufferList(AbsReplayBuffer):
    def __init__(self, config, initial_buffer=None):
        super().__init__(config, initial_buffer=initial_buffer)
        self.buffer = deque(maxlen=self.config.get('capacity', int(1e6)))

    def clear(self):
        super().clear()
        self.buffer = deque(maxlen=self.config.get('capacity', int(1e6)))

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        pickle.dump(self.buffer, open(os.path.join(save_dir, 'buffer.pkl'), 'wb'))

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        self.buffer = pickle.load(open(os.path.join(save_dir, 'buffer.pkl'), 'rb'))

    def reset_storage(self):
        self.clear()

    def push(self, item):
        if self.buffer.maxlen == len(self.buffer):
            disp = self.buffer[0]
        else:
            disp = None
        if self.config.get('tensor_tuple', False):
            # convert
            item = tuple(
                t if torch.is_tensor(t) else torch.tensor(t)
                for t in item
            )
        self.buffer.append(item)
        return disp

    def _grab_item_by_idx(self, idx):
        return self.buffer[idx]

    def sample_one(self):
        return self.buffer[torch.randint(0, self.__len__(), (1,))]

    def get_all(self):
        return self.buffer

    def __getitem__(self, item):
        if item >= self.__len__():
            raise IndexError
        return self._grab_item_by_idx(idx=item)

    def __len__(self):
        return len(self.buffer)


class ReplayBufferDiskStorage(AbsReplayBuffer):
    def __init__(self,
                 config,
                 initial_buffer=None
                 ):
        self.idx = 0
        self.size = 0
        self.capacity = config.get('capacity',int(1e6))
        self.device = config.get('device', None)
        storage_dir=config.get('storage_dir', None)
        if storage_dir is not None:
            self.set_storage_dir(storage_dir=storage_dir)
        super().__init__(config, initial_buffer=initial_buffer)

    def clear(self):
        super().clear()
        if self.storage_dir is not None:
            if os.path.exists(self.storage_dir):
                shutil.rmtree(self.storage_dir)

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        shutil.copytree(src=self.storage_dir, dst=save_dir)

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        self.clear()
        shutil.copytree(src=save_dir, dst=self.storage_dir)
        self.load_place(force=False)

    def set_storage_dir(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        self.reset_storage()

    def reset_storage(self):
        self.clear()
        os.makedirs(self.storage_dir)
        self.size = 0
        self.idx = 0
        self.save_place()

    def save_place(self):
        """
        saves idx and size to files as well
        """
        pickle.dump(
            {
                'size': self.size,
                'idx': self.idx,
            },
            open(self._get_file('info'), 'wb')
        )

    def load_place(self, force=False):
        info_file = self._get_file(name='info')
        if os.path.exists(info_file):
            dic = pickle.load(open(info_file, 'rb'))
            self.size = dic['size']
            self.idx = dic['idx']
        else:
            if force:
                print('failed to load file:', info_file)
                print('resetting storage')
                self.reset_storage()
            else:
                raise Exception('failed to load file: ' + info_file)

    def _get_file(self, name):
        return os.path.join(self.storage_dir, str(name) + '.pkl')

    def push(self, item):
        if self.size == self.capacity:
            disp = self.__getitem__(self.idx)
        else:
            disp = None
        pickle.dump(item, open(self._get_file(self.idx), 'wb'))

        self.size = max(self.idx + 1, self.size)
        self.idx = int((self.idx + 1) % self.capacity)

        self.save_place()
        return disp

    def _grab_item_by_idx(self, idx, change_device=True):
        item = pickle.load(open(self._get_file(name=idx), 'rb'))
        return self._convert_device(item=item, change_device=change_device)

    def _convert_device(self, item, change_device):
        if change_device:
            if type(item) == tuple:
                item = tuple(self._convert_device(t, change_device=change_device)
                             for t in item)
            elif torch.is_tensor(item):
                item = item.to(self.device)
        return item

    def sample_one(self):
        return self[torch.randint(0, self.size, (1,))]

    def __getitem__(self, item):
        if item >= self.size:
            raise IndexError
        return self._grab_item_by_idx(idx=int((self.idx + item) % self.size))

    def __len__(self):
        return self.size


if __name__ == '__main__':
    test = ReplayBufferDiskStorage(dict(capacity=3, storage_dir=os.path.join('replay_buffer_test')))
    test.extend('help')
    print([test[i] for i in range(len(test))])
    test.clear()

