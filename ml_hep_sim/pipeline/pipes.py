import os
import pickle

import torch
from treelib import Node, Tree

from ml_hep_sim.pipeline.pipeline_loggers import setup_logger


def pickle_save(path, name, obj):
    with open(path + name, "wb") as f:
        pickle.dump(obj, f)
    return obj


def pickle_load(path, name):
    with open(path + name, "rb") as f:
        obj = pickle.load(f)
    return obj


class Pipeline:
    def __init__(self, pipeline_name="pipe.p", pipeline_path="ml_pipeline/", logger=None):
        if pipeline_name[:-2] != ".p":
            pipeline_name += ".p"

        self.pipeline_name, self.pipeline_path = pipeline_name, pipeline_path

        if not os.path.exists(self.pipeline_path):
            os.makedirs(self.pipeline_path)

        if logger is None:
            self.logger = setup_logger(dummy_logger=True)
        else:
            self.logger = logger

        self.pipes = []

    def compose(self, *blocks):
        for b in blocks:
            if isinstance(b, self.__class__):
                self.pipes += b.pipes
            elif isinstance(b, list):
                self.pipes += b
            else:
                self.pipes.append(b)

        for b in self.pipes:
            b.logger = self.logger

        return self

    def fit(self, force=False, skip=False):
        for b, block in enumerate(self.pipes):
            self.logger.info(f"fitting #{b}: {block}!")
            if force:
                block._run()
            elif skip:
                continue
            else:
                if block.fit_block:
                    block._run()
                    block.fit_block = False
                else:
                    pass
                    # self.logger.warning(f"block {block} was already fitted!")
        return self

    def save(self):
        pickle_save(
            self.pipeline_path,
            self.pipeline_name,
            self.__dict__,
        )
        self.logger.info(f"Saving pipe to {self.pipeline_path + self.pipeline_name}")

        return self

    def load(self):
        saved_pipeline = pickle_load(
            self.pipeline_path,
            self.pipeline_name,
        )
        self.logger.info(f"Loading pipe from {self.pipeline_path + self.pipeline_name}")

        loaded_pipes = saved_pipeline["pipes"]

        if len(self.pipes) == len(loaded_pipes):
            for i in range(len(self.pipes)):
                self.pipes[i].__dict__.update(loaded_pipes[i].__dict__)
        else:
            self.logger.warning("Number of composed and loaded pipes did not match! Loading anyway...")
            self.__dict__.update(saved_pipeline)

        return self

    def _get_priors_recursively(self, block):
        priors = block.priors
        save = [block]

        if len(priors) == 0:
            return save

        for prior in priors:
            inner_priors = self._get_priors_recursively(prior)
            save.append(inner_priors)

        return save

    def draw_pipeline_tree(self, recursive_priors=None, block_idx=None, show_tree=False, to_graphviz_file=None):
        """https://stackoverflow.com/questions/64713797/visualizing-parse-tree-nested-list-to-tree-in-python"""

        if recursive_priors is None:
            recursive_priors = self._get_priors_recursively(self.pipes[-1 if block_idx is None else block_idx])

        root, *tail = recursive_priors
        tree = Tree()
        root_str = root.__class__.__name__ + "\n" + str(root.__repr__)[-15:-1]
        node = Node(root_str)
        tree.add_node(node)

        q = [[node, *tail]]
        while q:
            parent, *children = q.pop()
            for child in children:
                if isinstance(child, list):
                    head, *tail = child
                    head_str = head.__class__.__name__ + "\n" + str(head.__repr__)[-15:-1]
                    node = tree.create_node(head_str, parent=parent)
                    q.append([node, *tail])
                else:
                    child_str = child.__class__.__name__ + "\n" + str(child.__repr__)[-15:-1]
                    tree.create_node(child_str, parent=parent)

        if show_tree:
            print(tree.show())

        if to_graphviz_file:
            tree.to_graphviz(to_graphviz_file + ".dot", shape="box")

            try:
                os.system(f"dot -Tpng {to_graphviz_file}.dot > {to_graphviz_file}.png")
            except Exception as e:
                self.logger.error(e)
                self.logger.critical("Could not draw tree, need Graphviz library!")

        return tree

    def __add__(self, prior_pipeline):
        self.pipes += prior_pipeline.pipes
        return self


class Block:
    def __init__(self):
        self.priors = None  # initialized by __call__
        self.fit_block = True
        self.logger = None

    @staticmethod
    def _tensor_check(t):
        if torch.is_tensor(t):
            t = t.cpu().numpy()
        return t

    def update_current_block(self, *prior_blocks):
        """Method for parameter sharing between this (current) block and prior blocks.

        The idea is to update the values of None valued parameters in the current block to the values of the same but
        not None parameters in the prior blocks. Changed parameters need to be present in both compared objects.

        Note
        ----
        - Subtle point: prior values should always remain constant and should never be changed because they get overriden
        in the dict update. Only exception is the `results` attrtibute that gets accumulated in a list.
        - Prior blocks are represented by Block objects.

        Returns
        -------
        self

        """

        prior_dict = dict()
        for d in prior_blocks:
            for key, val in d.__dict__.items():

                # accumulate results as result list -> results from blocks get transformed to list!
                if key == "results":
                    if key not in prior_dict:
                        prior_dict[key] = list()
                    prior_dict[key].append(val)
                else:
                    prior_dict[key] = val

        for k, v in self.__dict__.items():
            if v is None and k in prior_dict:
                self.__dict__[k] = prior_dict[k]

        return self

    def run(self):
        """Needs to be implemented by every Block object."""
        pass

    def _run(self):
        """Internal run method that sets all the prior parameters and then executes the true run method."""
        self.update_current_block(*self.priors)
        self.run()
        return self

    def __call__(self, *priors):
        """Set prior Block objects."""
        self.priors = priors
        return self
