"""Turning a downloaded product into the crops its features keep of it.

Everything between the bytes on disk and the arrays ready to be written: each
instrument reads its own product, is placed against one feature, cut to it, and
written down. What turned out the same for every instrument lives in `common`.
"""
